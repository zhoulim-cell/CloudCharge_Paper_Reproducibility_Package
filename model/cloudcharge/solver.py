from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Callable, Optional

import numpy as np
from scipy.optimize import brentq, least_squares

from .attachment import AttachmentCoefficients, calculate_attachment_coefficients
from .atmosphere import AtmosphereState, build_atmosphere
from .config import SimulationConfig
from .constants import BOLTZMANN, ELEMENTARY_CHARGE, EPSILON_0, PI
from .profiles import cloud_shape, droplet_radius_at_time, peak_droplet_number


N_POS, N_NEG, S_POS, S_NEG, P_DROP = range(5)
STATE_NAMES = ("n_positive", "n_negative", "s_positive", "s_negative", "droplet_charge_e")


class SimulationCancelled(RuntimeError):
    pass


@dataclass
class FieldSolution:
    state: np.ndarray
    electric_field_v_m: np.ndarray
    dfield_dz_v_m2: np.ndarray
    charge_density_c_m3: np.ndarray
    current_density_a_m2: np.ndarray
    projection_iterations: int
    relative_current_residual: float


@dataclass
class SimulationResult:
    z_m: np.ndarray
    snapshot_times_s: np.ndarray
    snapshots: dict[float, dict[str, np.ndarray | float]]
    diagnostics: dict[str, np.ndarray | float | str]
    metadata: dict[str, object]

    def nearest_snapshot_time(self, time_s: float) -> float:
        times = np.asarray(self.snapshot_times_s, dtype=float)
        return float(times[np.argmin(np.abs(times - time_s))])


ProgressCallback = Callable[[float, str], None]
CancelCallback = Callable[[], bool]


def _bernoulli_function(x: np.ndarray) -> np.ndarray:
    """Stable B(x)=x/(exp(x)-1), including large positive/negative x."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    small = np.abs(x) < 1.0e-5
    xs = x[small]
    out[small] = 1.0 - xs / 2.0 + xs**2 / 12.0 - xs**4 / 720.0

    large_pos = x > 50.0
    out[large_pos] = x[large_pos] * np.exp(-x[large_pos])

    large_neg = x < -50.0
    out[large_neg] = -x[large_neg]

    regular = ~(small | large_pos | large_neg)
    out[regular] = x[regular] / np.expm1(x[regular])
    return out


def _gradient_corrected(values: np.ndarray, z: np.ndarray) -> np.ndarray:
    return np.gradient(values, z, edge_order=2)


def _gradient_legacy(values: np.ndarray, dz: float) -> np.ndarray:
    """Reproduce the principal derivative stencil in the uploaded programs."""
    n = values.size
    out = np.zeros_like(values)
    if n < 5:
        return np.gradient(values, dz, edge_order=1)
    out[2:-2] = (
        values[4:] + values[3:-1] - values[1:-3] - values[:-4]
    ) / (6.0 * dz)
    out[1] = (values[2] - values[0]) / (4.0 * dz)
    out[-2] = (values[-1] - values[-3]) / (4.0 * dz)
    out[0] = (values[0] - values[1]) / (2.0 * dz)
    out[-1] = (values[-1] - values[-2]) / (2.0 * dz)
    return out


def _laplacian_no_flux(values: np.ndarray, dz: float) -> np.ndarray:
    out = np.empty_like(values)
    out[1:-1] = (values[:-2] - 2.0 * values[1:-1] + values[2:]) / dz**2
    out[0] = 2.0 * (values[1] - values[0]) / dz**2
    out[-1] = 2.0 * (values[-2] - values[-1]) / dz**2
    return out


def _cumulative_trapezoid(values: np.ndarray, z: np.ndarray) -> np.ndarray:
    out = np.zeros_like(values)
    out[1:] = np.cumsum(0.5 * (values[1:] + values[:-1]) * np.diff(z))
    return out


def _conservative_filter(values: np.ndarray, strength: float) -> np.ndarray:
    if strength <= 0.0:
        return values
    lam = min(float(strength), 0.24)
    flux = lam * np.diff(values)
    out = values.copy()
    out[:-1] += flux
    out[1:] -= flux
    return out


class CloudChargeModel:
    def __init__(self, config: SimulationConfig):
        config.validate()
        self.config = config
        self.atmosphere: AtmosphereState = build_atmosphere(config.atmosphere)
        self.attachment: AttachmentCoefficients = calculate_attachment_coefficients(
            self.atmosphere,
            config.aerosol.mean_radius_m,
            config.physics.attachment_mode,
        )
        self.z = np.linspace(
            0.0, config.domain.length_m, config.domain.grid_points, dtype=float
        )
        self.dz = float(self.z[1] - self.z[0])
        self.cloud_shape = cloud_shape(self.z, config.domain)
        self.droplet_peak_m3 = peak_droplet_number(config.droplet)
        self.droplet_number_m3 = self.droplet_peak_m3 * self.cloud_shape
        self.total_aerosol_m3 = np.full_like(self.z, config.aerosol.total_number_m3)
        self._clip_count = 0

    @property
    def beta_pos(self) -> float:
        return self.attachment.beta_positive_m3_s

    @property
    def beta_neg(self) -> float:
        return self.attachment.beta_negative_m3_s

    @property
    def gamma_pos(self) -> float:
        return self.attachment.gamma_positive_m3_s

    @property
    def gamma_neg(self) -> float:
        return self.attachment.gamma_negative_m3_s

    def _homogeneous_equilibrium_corrected(self) -> tuple[float, float, float, float]:
        q = self.config.atmosphere.ion_pair_production_m3_s
        alpha = self.config.atmosphere.recombination_m3_s
        st = self.config.aerosol.total_number_m3
        n_scale = math.sqrt(q / alpha)

        def species_from_n(n_pos: float, n_neg: float) -> tuple[float, float, float]:
            ratio_pos = self.beta_pos * n_pos / max(self.gamma_neg * n_neg, 1.0e-300)
            ratio_neg = self.beta_neg * n_neg / max(self.gamma_pos * n_pos, 1.0e-300)
            neutral = st / (1.0 + ratio_pos + ratio_neg)
            s_pos = ratio_pos * neutral
            s_neg = ratio_neg * neutral
            return neutral, s_pos, s_neg

        def residual(log_n: np.ndarray) -> np.ndarray:
            n_pos, n_neg = np.exp(log_n)
            neutral, s_pos, s_neg = species_from_n(n_pos, n_neg)
            balance = q - alpha * n_pos * n_neg - neutral * (
                self.beta_pos * n_pos + self.beta_neg * n_neg
            )
            charge = n_pos - n_neg + s_pos - s_neg
            return np.array([balance / q, charge / n_scale])

        guess = np.log([0.7 * n_scale, 0.7 * n_scale])
        solution = least_squares(
            residual,
            guess,
            xtol=1.0e-12,
            ftol=1.0e-12,
            gtol=1.0e-12,
            max_nfev=1000,
        )
        if not solution.success:
            raise RuntimeError(f"Initial equilibrium solve failed: {solution.message}")
        n_pos, n_neg = np.exp(solution.x)
        _, s_pos, s_neg = species_from_n(float(n_pos), float(n_neg))
        return float(n_pos), float(n_neg), float(s_pos), float(s_neg)

    def _homogeneous_equilibrium_legacy(self) -> tuple[float, float, float, float]:
        q = self.config.atmosphere.ion_pair_production_m3_s
        alpha = self.config.atmosphere.recombination_m3_s
        st = self.config.aerosol.total_number_m3
        beta_mean = 0.5 * (self.beta_pos + self.beta_neg)
        # Deliberately reproduces the dimensionally inconsistent initialization in the IDL code.
        n0 = (-beta_mean + math.sqrt(beta_mean**2 + q * alpha)) / alpha

        def residual(x: float) -> float:
            root = math.sqrt(1.0 + (x / 2.0) ** 2)
            n_pos_ratio = root + x / 2.0
            n_neg_ratio = root - x / 2.0
            ratio_pos = self.beta_pos * n_pos_ratio / (self.gamma_neg * n_neg_ratio)
            ratio_neg = self.beta_neg * n_neg_ratio / (self.gamma_pos * n_pos_ratio)
            neutral = st / (1.0 + ratio_pos + ratio_neg)
            return n0 * x + neutral * (ratio_pos - ratio_neg)

        try:
            x = float(brentq(residual, -1.0, 1.0, maxiter=200))
        except ValueError:
            x = 0.0
        root = math.sqrt(1.0 + (x / 2.0) ** 2)
        n_pos = n0 * (root + x / 2.0)
        n_neg = n0 * (root - x / 2.0)
        ratio_pos = self.beta_pos * n_pos / (self.gamma_neg * n_neg)
        ratio_neg = self.beta_neg * n_neg / (self.gamma_pos * n_pos)
        neutral = st / (1.0 + ratio_pos + ratio_neg)
        return n_pos, n_neg, neutral * ratio_pos, neutral * ratio_neg

    def initial_state(self) -> np.ndarray:
        if self.config.physics.use_legacy_initialization:
            n_pos, n_neg, s_pos, s_neg = self._homogeneous_equilibrium_legacy()
        else:
            n_pos, n_neg, s_pos, s_neg = self._homogeneous_equilibrium_corrected()
        state = np.zeros((5, self.z.size), dtype=float)
        state[N_POS] = n_pos
        state[N_NEG] = n_neg
        state[S_POS] = s_pos
        state[S_NEG] = s_neg
        state[P_DROP] = 0.0
        return self._enforce_bounds(state)

    def _gradient(self, values: np.ndarray) -> np.ndarray:
        if self.config.physics.use_legacy_initialization:
            return _gradient_legacy(values, self.dz)
        return _gradient_corrected(values, self.z)

    def _charge_density(self, state: np.ndarray) -> np.ndarray:
        return ELEMENTARY_CHARGE * (
            state[N_POS]
            - state[N_NEG]
            + state[S_POS]
            - state[S_NEG]
            + state[P_DROP] * self.droplet_number_m3
        )

    def _poisson_field(
        self, state: np.ndarray, charge_density: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        dfield_dz = charge_density / EPSILON_0
        integral = _cumulative_trapezoid(dfield_dz, self.z)
        mu_pos = self.atmosphere.positive_mobility_m2_v_s
        mu_neg = self.atmosphere.negative_mobility_m2_v_s
        conductivity_number = mu_pos * state[N_POS] + mu_neg * state[N_NEG]
        conductivity_number = np.maximum(conductivity_number, 1.0e-30)
        j = self.config.atmosphere.current_density_a_m2
        e_cond = j / (ELEMENTARY_CHARGE * conductivity_number)

        anchor = self.config.physics.field_anchor
        if anchor == "bottom":
            offset = e_cond[0]
        elif anchor == "top":
            offset = e_cond[-1] - integral[-1]
        else:
            weights = (ELEMENTARY_CHARGE * conductivity_number) ** 2
            offset = float(np.sum(weights * (e_cond - integral)) / np.sum(weights))
        return integral + offset, dfield_dz

    def solve_field(self, state: np.ndarray) -> FieldSolution:
        state = state.copy()
        mu_pos = self.atmosphere.positive_mobility_m2_v_s
        mu_neg = self.atmosphere.negative_mobility_m2_v_s
        j_target = self.config.atmosphere.current_density_a_m2
        mode = self.config.physics.field_mode
        projection_iterations = 0

        if mode == "conductivity_only":
            sigma_number = np.maximum(mu_pos * state[N_POS] + mu_neg * state[N_NEG], 1.0e-30)
            electric_field = j_target / (ELEMENTARY_CHARGE * sigma_number)
            dfield_dz = self._gradient(electric_field)
            charge_density = self._charge_density(state)
            current = ELEMENTARY_CHARGE * sigma_number * electric_field
            residual = float(
                np.max(np.abs(current - j_target)) / max(abs(j_target), 1.0e-30)
            )
            return FieldSolution(
                state,
                electric_field,
                dfield_dz,
                charge_density,
                current,
                projection_iterations,
                residual,
            )

        for iteration in range(max(1, self.config.numerics.field_iterations)):
            charge_density = self._charge_density(state)
            electric_field, dfield_dz = self._poisson_field(state, charge_density)
            sigma_number = mu_pos * state[N_POS] + mu_neg * state[N_NEG]
            current = ELEMENTARY_CHARGE * sigma_number * electric_field
            denom = max(abs(j_target), 1.0e-30)
            relative = float(np.max(np.abs(current - j_target)) / denom)
            projection_iterations = iteration + 1

            if mode == "poisson_only" or abs(j_target) < 1.0e-30:
                break
            if relative <= self.config.numerics.field_tolerance:
                break

            valid = np.abs(electric_field) > 1.0e-12
            target_sigma = np.zeros_like(electric_field)
            target_sigma[valid] = j_target / (
                ELEMENTARY_CHARGE * electric_field[valid]
            )
            valid &= target_sigma > 0.0
            delta = np.zeros_like(electric_field)
            delta[valid] = (target_sigma[valid] - sigma_number[valid]) / (
                mu_pos + mu_neg
            )

            # Adding the same delta to n+ and n- preserves space charge while adjusting conductivity.
            lower = -0.95 * np.minimum(state[N_POS], state[N_NEG])
            upper = 5.0 * np.maximum(state[N_POS], state[N_NEG])
            delta = np.clip(delta, lower, upper)
            delta *= self.config.numerics.field_relaxation
            state[N_POS] += delta
            state[N_NEG] += delta
            state = self._enforce_bounds(state)

        charge_density = self._charge_density(state)
        electric_field, dfield_dz = self._poisson_field(state, charge_density)
        sigma_number = mu_pos * state[N_POS] + mu_neg * state[N_NEG]
        current = ELEMENTARY_CHARGE * sigma_number * electric_field
        residual = float(
            np.max(np.abs(current - j_target)) / max(abs(j_target), 1.0e-30)
        )
        return FieldSolution(
            state,
            electric_field,
            dfield_dz,
            charge_density,
            current,
            projection_iterations,
            residual,
        )

    def _charging_terms(
        self,
        state: np.ndarray,
        electric_field: np.ndarray,
        time_s: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray | float]]:
        n_pos = state[N_POS]
        n_neg = state[N_NEG]
        p_drop = state[P_DROP]
        nd = self.droplet_number_m3
        radius = droplet_radius_at_time(time_s, self.config.droplet)
        temperature = self.atmosphere.temperature_k
        d_pos = self.atmosphere.positive_diffusivity_m2_s
        d_neg = self.atmosphere.negative_diffusivity_m2_s

        if self.config.physics.charging_mode == "revised":
            x = p_drop * ELEMENTARY_CHARGE**2 / (
                4.0 * PI * EPSILON_0 * radius * BOLTZMANN * temperature
            )
            b_pos = _bernoulli_function(x)
            b_neg = _bernoulli_function(-x)
            diffusion_pos = 4.0 * PI * d_pos * n_pos * nd * radius * b_pos
            diffusion_neg = 4.0 * PI * d_neg * n_neg * nd * radius * b_neg
        else:
            ratio = np.maximum(d_pos * n_pos, 1.0e-300) / np.maximum(
                d_neg * n_neg, 1.0e-300
            )
            x = np.log(ratio)
            b_common = _bernoulli_function(x)
            diffusion_pos = 4.0 * PI * d_pos * n_pos * nd * radius * b_common
            diffusion_neg = 4.0 * PI * d_neg * n_neg * nd * radius * b_common

        diffusion_pos = np.maximum(diffusion_pos, 0.0)
        diffusion_neg = np.maximum(diffusion_neg, 0.0)
        drift_pos = np.zeros_like(n_pos)
        drift_neg = np.zeros_like(n_neg)

        if self.config.physics.include_drift_charging:
            if self.config.physics.charging_mode == "revised":
                factor = ELEMENTARY_CHARGE * np.abs(electric_field) * radius / (
                    2.0 * BOLTZMANN * temperature
                )
                drift_pos = diffusion_pos * factor
                drift_neg = diffusion_neg * factor
            else:
                denominator = 3.0 * electric_field * radius**2
                valid = np.abs(denominator) > 1.0e-30
                image = p_drop * ELEMENTARY_CHARGE / (4.0 * PI * EPSILON_0)
                drift_pos[valid] = (
                    PI
                    * n_pos[valid]
                    * self.atmosphere.positive_mobility_m2_v_s
                    * nd[valid]
                    * (3.0 * electric_field[valid] * radius**2 - image[valid]) ** 2
                    / denominator[valid]
                )
                drift_neg[valid] = (
                    PI
                    * n_neg[valid]
                    * self.atmosphere.negative_mobility_m2_v_s
                    * nd[valid]
                    * (3.0 * electric_field[valid] * radius**2 + image[valid]) ** 2
                    / denominator[valid]
                )
                drift_pos = np.maximum(drift_pos, 0.0)
                drift_neg = np.maximum(drift_neg, 0.0)

        charge_rate = np.zeros_like(p_drop)
        cloudy = nd > 0.0
        charge_rate[cloudy] = (
            diffusion_pos[cloudy]
            + drift_pos[cloudy]
            - diffusion_neg[cloudy]
            - drift_neg[cloudy]
        ) / nd[cloudy]

        details: dict[str, np.ndarray | float] = {
            "droplet_radius_m": radius,
            "charging_dimensionless": x,
            "droplet_diffusion_loss_positive_m3_s": diffusion_pos,
            "droplet_diffusion_loss_negative_m3_s": diffusion_neg,
            "droplet_drift_loss_positive_m3_s": drift_pos,
            "droplet_drift_loss_negative_m3_s": drift_neg,
            "droplet_charge_rate_e_s": charge_rate,
        }
        return diffusion_pos + drift_pos, diffusion_neg + drift_neg, charge_rate, details

    def rhs(
        self,
        state: np.ndarray,
        time_s: float,
        field: FieldSolution,
        return_terms: bool = False,
    ) -> tuple[np.ndarray, Optional[dict[str, np.ndarray | float]]]:
        n_pos = state[N_POS]
        n_neg = state[N_NEG]
        s_pos = state[S_POS]
        s_neg = state[S_NEG]
        neutral = np.maximum(self.total_aerosol_m3 - s_pos - s_neg, 0.0)
        efield = field.electric_field_v_m
        dfield_dz = field.dfield_dz_v_m2

        grad_n_pos = self._gradient(n_pos)
        grad_n_neg = self._gradient(n_neg)
        drift_div_pos = self.atmosphere.positive_mobility_m2_v_s * (
            n_pos * dfield_dz + efield * grad_n_pos
        )
        drift_div_neg = self.atmosphere.negative_mobility_m2_v_s * (
            n_neg * dfield_dz + efield * grad_n_neg
        )

        aerosol_loss_pos = self.beta_pos * neutral * n_pos + self.gamma_pos * s_neg * n_pos
        aerosol_loss_neg = self.beta_neg * neutral * n_neg + self.gamma_neg * s_pos * n_neg
        droplet_loss_pos, droplet_loss_neg, charge_rate, charge_details = self._charging_terms(
            state, efield, time_s
        )

        q = self.config.atmosphere.ion_pair_production_m3_s
        alpha = self.config.atmosphere.recombination_m3_s
        recombination = alpha * n_pos * n_neg

        derivative = np.zeros_like(state)
        derivative[N_POS] = q - recombination - aerosol_loss_pos - drift_div_pos - droplet_loss_pos
        derivative[N_NEG] = q - recombination - aerosol_loss_neg + drift_div_neg - droplet_loss_neg
        derivative[S_POS] = self.beta_pos * neutral * n_pos - self.gamma_neg * s_pos * n_neg
        derivative[S_NEG] = self.beta_neg * neutral * n_neg - self.gamma_pos * s_neg * n_pos
        derivative[P_DROP] = charge_rate

        if self.config.physics.include_ion_diffusion:
            derivative[N_POS] += self.atmosphere.positive_diffusivity_m2_s * _laplacian_no_flux(
                n_pos, self.dz
            )
            derivative[N_NEG] += self.atmosphere.negative_diffusivity_m2_s * _laplacian_no_flux(
                n_neg, self.dz
            )

        terms = None
        if return_terms:
            terms = {
                "neutral_aerosol_m3": neutral,
                "recombination_m3_s": recombination,
                "aerosol_loss_positive_m3_s": aerosol_loss_pos,
                "aerosol_loss_negative_m3_s": aerosol_loss_neg,
                "ion_drift_divergence_positive_m3_s": drift_div_pos,
                "ion_drift_divergence_negative_m3_s": drift_div_neg,
                **charge_details,
            }
        return derivative, terms

    def _enforce_bounds(self, state: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=float).copy()
        before = state.copy()
        state[N_POS] = np.maximum(state[N_POS], 1.0)
        state[N_NEG] = np.maximum(state[N_NEG], 1.0)
        state[S_POS] = np.maximum(state[S_POS], 0.0)
        state[S_NEG] = np.maximum(state[S_NEG], 0.0)

        charged_sum = state[S_POS] + state[S_NEG]
        limit = 0.999999 * self.total_aerosol_m3
        over = charged_sum > limit
        if np.any(over):
            scale = limit[over] / charged_sum[over]
            state[S_POS, over] *= scale
            state[S_NEG, over] *= scale

        state[P_DROP] = np.clip(
            state[P_DROP],
            -self.config.droplet.max_abs_charge_e,
            self.config.droplet.max_abs_charge_e,
        )
        state[P_DROP, self.droplet_number_m3 <= 0.0] = 0.0
        if not np.all(np.isfinite(state)):
            raise FloatingPointError("Non-finite state encountered during integration.")
        self._clip_count += int(np.count_nonzero(state != before))
        return state

    def _apply_conservative_smoothing(self, state: np.ndarray) -> np.ndarray:
        strength = self.config.numerics.conservative_smoothing
        if strength <= 0.0:
            return state
        out = state.copy()
        for index in (N_POS, N_NEG, S_POS, S_NEG):
            out[index] = _conservative_filter(out[index], strength)
        charge_per_volume = out[P_DROP] * self.droplet_number_m3
        charge_per_volume = _conservative_filter(charge_per_volume, strength)
        cloudy = self.droplet_number_m3 > 0.0
        out[P_DROP, cloudy] = charge_per_volume[cloudy] / self.droplet_number_m3[cloudy]
        out[P_DROP, ~cloudy] = 0.0
        return self._enforce_bounds(out)

    def _positivity_limited_dt(self, state: np.ndarray, derivative: np.ndarray, proposed: float) -> float:
        allowed = proposed
        safety = self.config.numerics.positivity_safety
        for index in (N_POS, N_NEG, S_POS, S_NEG):
            negative = derivative[index] < 0.0
            if np.any(negative):
                local = -safety * state[index, negative] / derivative[index, negative]
                allowed = min(allowed, float(np.min(local)))
        return max(self.config.numerics.min_dt_s, min(proposed, allowed))

    def _snapshot(self, state: np.ndarray, time_s: float, field: FieldSolution) -> dict[str, np.ndarray | float]:
        neutral = np.maximum(self.total_aerosol_m3 - state[S_POS] - state[S_NEG], 0.0)
        return {
            "time_s": float(time_s),
            "n_positive_m3": state[N_POS].copy(),
            "n_negative_m3": state[N_NEG].copy(),
            "s_positive_m3": state[S_POS].copy(),
            "s_negative_m3": state[S_NEG].copy(),
            "neutral_aerosol_m3": neutral.copy(),
            "droplet_charge_e": state[P_DROP].copy(),
            "droplet_number_m3": self.droplet_number_m3.copy(),
            "droplet_radius_m": float(droplet_radius_at_time(time_s, self.config.droplet)),
            "electric_field_v_m": field.electric_field_v_m.copy(),
            "dfield_dz_v_m2": field.dfield_dz_v_m2.copy(),
            "charge_density_c_m3": field.charge_density_c_m3.copy(),
            "current_density_a_m2": field.current_density_a_m2.copy(),
        }

    def run(
        self,
        progress_callback: Optional[ProgressCallback] = None,
        cancel_callback: Optional[CancelCallback] = None,
    ) -> SimulationResult:
        if self.config.physics.solution_mode == "paper2007_bvp":
            from .paper2007 import run_paper2007_equilibrium

            return run_paper2007_equilibrium(
                self,
                progress_callback=progress_callback,
                cancel_callback=cancel_callback,
            )
        if self.config.physics.solution_mode == "paper2012_time":
            from .paper2012 import run_paper2012_time_dependent

            return run_paper2012_time_dependent(
                self,
                progress_callback=progress_callback,
                cancel_callback=cancel_callback,
            )
        if self.config.physics.solution_mode == "paper2012_direct":
            from .paper2012_direct import run_paper2012_direct

            return run_paper2012_direct(
                self,
                progress_callback=progress_callback,
                cancel_callback=cancel_callback,
            )
        if self.config.physics.solution_mode == "paper2012_spectrum":
            from .spectrum import run_paper2012_spectrum

            return run_paper2012_spectrum(
                self,
                progress_callback=progress_callback,
                cancel_callback=cancel_callback,
            )

        self._clip_count = 0
        duration = self.config.numerics.duration_s
        snapshots_requested = sorted(
            set(
                [0.0, duration]
                + [
                    min(duration, max(0.0, float(t)))
                    for t in self.config.numerics.snapshot_times_s
                ]
            )
        )
        state = self.initial_state()
        field = self.solve_field(state)
        state = field.state
        time_s = 0.0
        snapshots: dict[float, dict[str, np.ndarray | float]] = {
            0.0: self._snapshot(state, 0.0, field)
        }
        next_snapshot_index = 1

        diag_time: list[float] = []
        diag_max_field: list[float] = []
        diag_max_charge: list[float] = []
        diag_current_residual: list[float] = []
        diag_net_charge: list[float] = []
        next_diag_time = 0.0
        diag_interval = max(duration / 250.0, self.config.numerics.max_dt_s)
        step_count = 0
        min_dt_used = self.config.numerics.max_dt_s
        max_dt_used = 0.0

        def record_diagnostics() -> None:
            diag_time.append(time_s)
            diag_max_field.append(float(np.max(np.abs(field.electric_field_v_m))))
            diag_max_charge.append(float(np.max(np.abs(state[P_DROP]))))
            diag_current_residual.append(field.relative_current_residual)
            diag_net_charge.append(float(np.sum(0.5 * (field.charge_density_c_m3[1:] + field.charge_density_c_m3[:-1]) * np.diff(self.z))))

        record_diagnostics()
        if progress_callback:
            progress_callback(0.0, "初始化完成，开始积分")

        while time_s < duration - 1.0e-12:
            if cancel_callback and cancel_callback():
                raise SimulationCancelled("Simulation cancelled by user.")

            field = self.solve_field(state)
            state = field.state
            rhs1, _ = self.rhs(state, time_s, field)

            proposed_dt = min(self.config.numerics.max_dt_s, duration - time_s)
            if next_snapshot_index < len(snapshots_requested):
                proposed_dt = min(
                    proposed_dt, snapshots_requested[next_snapshot_index] - time_s
                )
            dt = self._positivity_limited_dt(state, rhs1, proposed_dt)
            if dt > proposed_dt:
                dt = proposed_dt
            if dt <= 0.0:
                raise RuntimeError("Non-positive time step encountered.")

            stage1 = self._enforce_bounds(state + dt * rhs1)
            stage1_field = self.solve_field(stage1)
            stage1 = stage1_field.state
            rhs2, _ = self.rhs(stage1, time_s + dt, stage1_field)
            state = self._enforce_bounds(0.5 * state + 0.5 * (stage1 + dt * rhs2))
            state = self._apply_conservative_smoothing(state)
            field = self.solve_field(state)
            state = field.state

            time_s += dt
            step_count += 1
            min_dt_used = min(min_dt_used, dt)
            max_dt_used = max(max_dt_used, dt)

            while (
                next_snapshot_index < len(snapshots_requested)
                and time_s >= snapshots_requested[next_snapshot_index] - 1.0e-9
            ):
                snap_time = snapshots_requested[next_snapshot_index]
                snapshots[snap_time] = self._snapshot(state, snap_time, field)
                next_snapshot_index += 1

            if time_s >= next_diag_time - 1.0e-12:
                record_diagnostics()
                next_diag_time += diag_interval

            if progress_callback and (
                step_count == 1 or step_count % max(1, int(50.0 / max(dt, 1.0e-9))) == 0
            ):
                progress_callback(
                    min(1.0, time_s / duration),
                    f"t={time_s:.1f} s，max|E|={np.max(np.abs(field.electric_field_v_m)):.2f} V/m",
                )

        final_field = self.solve_field(state)
        state = final_field.state
        _, final_terms = self.rhs(state, duration, final_field, return_terms=True)
        snapshots[duration] = self._snapshot(state, duration, final_field)
        if not diag_time or abs(diag_time[-1] - duration) > 1.0e-9:
            field = final_field
            time_s = duration
            record_diagnostics()

        poisson_numeric = self._gradient(final_field.electric_field_v_m)
        poisson_residual = poisson_numeric - final_field.charge_density_c_m3 / EPSILON_0
        diagnostics: dict[str, np.ndarray | float | str] = {
            "time_s": np.asarray(diag_time),
            "max_abs_electric_field_v_m": np.asarray(diag_max_field),
            "max_abs_droplet_charge_e": np.asarray(diag_max_charge),
            "relative_current_residual": np.asarray(diag_current_residual),
            "column_net_charge_c_m2": np.asarray(diag_net_charge),
            "final_poisson_residual_v_m2": poisson_residual,
            "final_max_relative_current_residual": final_field.relative_current_residual,
            "time_steps": float(step_count),
            "min_dt_s": float(min_dt_used),
            "max_dt_s": float(max_dt_used),
            "bound_clip_count": float(self._clip_count),
        }
        if final_terms:
            diagnostics.update({f"final_{k}": v for k, v in final_terms.items()})

        metadata: dict[str, object] = {
            "config": self.config.to_dict(),
            "atmosphere": asdict(self.atmosphere),
            "attachment": {
                "beta_positive_m3_s": self.beta_pos,
                "beta_negative_m3_s": self.beta_neg,
                "gamma_positive_m3_s": self.gamma_pos,
                "gamma_negative_m3_s": self.gamma_neg,
                **self.attachment.metadata,
            },
            "droplet_peak_number_m3": self.droplet_peak_m3,
            "model_notes": (
                "One-dimensional prescribed cloud-edge profile; small ions are transported, "
                "aerosols and droplets are stationary except for charging."
            ),
        }
        if progress_callback:
            progress_callback(1.0, "模拟完成")
        return SimulationResult(
            z_m=self.z.copy(),
            snapshot_times_s=np.asarray(sorted(snapshots.keys()), dtype=float),
            snapshots=snapshots,
            diagnostics=diagnostics,
            metadata=metadata,
        )
