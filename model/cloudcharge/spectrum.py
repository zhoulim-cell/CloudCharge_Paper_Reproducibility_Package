from __future__ import annotations

from dataclasses import asdict
import math
from typing import Optional

import numpy as np
from scipy.special import ndtr

from .constants import BOLTZMANN, ELEMENTARY_CHARGE, EPSILON_0, PI
from .paper2012 import _equilibrium_result, _estimate_tau
from .solver import (
    N_NEG,
    N_POS,
    P_DROP,
    S_NEG,
    S_POS,
    CloudChargeModel,
    FieldSolution,
    SimulationCancelled,
    SimulationResult,
    _laplacian_no_flux,
)
try:
    from numba import njit, prange
    _HAS_NUMBA = True
except Exception:  # pragma: no cover - optional acceleration
    _HAS_NUMBA = False
    njit = None
    prange = range


if _HAS_NUMBA:
    @njit(cache=True)
    def _implicit_birth_death_numba(spectrum, rate_pos, rate_neg, dt):
        nz, nr, nq = spectrum.shape
        batch = nz * nr
        rhs = spectrum.reshape((batch, nq))
        rp = rate_pos.reshape((batch, nq))
        rn = rate_neg.reshape((batch, nq))
        cprime = np.empty((batch, nq), dtype=np.float64)
        dprime = np.empty((batch, nq), dtype=np.float64)
        result = np.empty((batch, nq), dtype=np.float64)
        for b in range(batch):
            main0 = 1.0 + dt * (rp[b, 0] + rn[b, 0])
            if main0 < 1.0e-300:
                main0 = 1.0e-300
            upper0 = -dt * rn[b, 1] if nq > 1 else 0.0
            cprime[b, 0] = upper0 / main0
            dprime[b, 0] = rhs[b, 0] / main0
            for k in range(1, nq):
                lower = -dt * rp[b, k - 1]
                main = 1.0 + dt * (rp[b, k] + rn[b, k])
                denom = main - lower * cprime[b, k - 1]
                if denom < 1.0e-300:
                    denom = 1.0e-300
                upper = -dt * rn[b, k + 1] if k < nq - 1 else 0.0
                cprime[b, k] = upper / denom
                dprime[b, k] = (rhs[b, k] - lower * dprime[b, k - 1]) / denom
            result[b, nq - 1] = dprime[b, nq - 1]
            for k in range(nq - 2, -1, -1):
                result[b, k] = dprime[b, k] - cprime[b, k] * result[b, k + 1]
        return result.reshape((nz, nr, nq))



def _bernoulli(x: np.ndarray) -> np.ndarray:
    """Stable Bernoulli function B(x)=x/(exp(x)-1)."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    small = np.abs(x) < 1.0e-5
    xs = x[small]
    out[small] = 1.0 - xs / 2.0 + xs * xs / 12.0 - xs**4 / 720.0
    regular = ~small
    xr = x[regular]
    values = np.empty_like(xr)
    large_pos = xr > 50.0
    large_neg = xr < -50.0
    middle = ~(large_pos | large_neg)
    values[large_pos] = xr[large_pos] * np.exp(-xr[large_pos])
    values[large_neg] = -xr[large_neg]
    values[middle] = xr[middle] / np.expm1(xr[middle])
    out[regular] = values
    return out


def _radius_grid(model: CloudChargeModel) -> tuple[np.ndarray, np.ndarray]:
    cfg = model.config.spectrum
    if cfg.radius_spacing == "log":
        edges = np.geomspace(cfg.radius_min_m, cfg.radius_max_m, cfg.radius_bins + 1)
        centers = np.sqrt(edges[:-1] * edges[1:])
    else:
        edges = np.linspace(cfg.radius_min_m, cfg.radius_max_m, cfg.radius_bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
    return centers.astype(float), edges.astype(float)


def _size_bin_weights(model: CloudChargeModel, edges: np.ndarray) -> np.ndarray:
    cfg = model.config.spectrum
    r0 = model.config.droplet.initial_radius_m
    if cfg.initial_size_distribution == "monodisperse":
        centers = 0.5 * (edges[:-1] + edges[1:])
        weights = np.zeros(centers.size, dtype=float)
        weights[int(np.argmin(np.abs(centers - r0)))] = 1.0
        return weights
    sigma = math.log(cfg.geometric_std)
    z_hi = (np.log(edges[1:]) - math.log(r0)) / sigma
    z_lo = (np.log(edges[:-1]) - math.log(r0)) / sigma
    weights = ndtr(z_hi) - ndtr(z_lo)
    if float(np.sum(weights)) <= 0:
        raise ValueError("Initial droplet radius lies outside the spectral radius grid.")
    return weights / np.sum(weights)


def _charge_bin_weights(model: CloudChargeModel, charges: np.ndarray) -> np.ndarray:
    cfg = model.config.spectrum
    if cfg.initial_charge_mode == "neutral":
        weights = np.zeros(charges.size, dtype=float)
        weights[int(np.argmin(np.abs(charges)))] = 1.0
        return weights
    weights = np.exp(-0.5 * (charges / cfg.initial_charge_std_e) ** 2)
    return weights / np.sum(weights)


def _moments(
    spectrum: np.ndarray, radius_m: np.ndarray, charges_e: np.ndarray
) -> dict[str, np.ndarray]:
    number = np.sum(spectrum, axis=(1, 2))
    safe = np.maximum(number, 1.0e-300)
    size_number = np.sum(spectrum, axis=2)
    charge_number = np.sum(spectrum, axis=1)
    charge_first = np.sum(charge_number * charges_e[None, :], axis=1)
    charge_second = np.sum(charge_number * charges_e[None, :] ** 2, axis=1)
    mean_charge = charge_first / safe
    variance_charge = np.maximum(charge_second / safe - mean_charge**2, 0.0)
    radius_first = np.sum(size_number * radius_m[None, :], axis=1)
    radius_second = np.sum(size_number * radius_m[None, :] ** 2, axis=1)
    mean_radius = radius_first / safe
    variance_radius = np.maximum(radius_second / safe - mean_radius**2, 0.0)
    positive = np.sum(charge_number[:, charges_e > 0], axis=1) / safe
    negative = np.sum(charge_number[:, charges_e < 0], axis=1) / safe
    neutral = np.sum(charge_number[:, charges_e == 0], axis=1) / safe
    mode_index = np.argmax(charge_number, axis=1)
    modal_charge = charges_e[mode_index]
    cross = np.sum(
        spectrum * radius_m[None, :, None] * charges_e[None, None, :], axis=(1, 2)
    ) / safe
    covariance = cross - mean_radius * mean_charge
    return {
        "number_m3": number,
        "size_number_m3": size_number,
        "charge_number_m3": charge_number,
        "mean_charge_e": mean_charge,
        "charge_std_e": np.sqrt(variance_charge),
        "mean_radius_m": mean_radius,
        "radius_std_m": np.sqrt(variance_radius),
        "positive_fraction": positive,
        "negative_fraction": negative,
        "neutral_fraction": neutral,
        "modal_charge_e": modal_charge.astype(float),
        "radius_charge_covariance_m_e": covariance,
        "charge_number_density_e_m3": charge_first,
    }


def _growth_transform(edges: np.ndarray, model: CloudChargeModel, t0: float, t1: float) -> np.ndarray:
    cfg = model.config
    sp = cfg.spectrum
    growth_time = cfg.droplet.growth_time_s
    if sp.growth_law == "none" or growth_time <= 0 or t0 >= growth_time:
        return edges.copy()
    a = min(max(t0, 0.0), growth_time) / growth_time
    b = min(max(t1, 0.0), growth_time) / growth_time
    if b <= a:
        return edges.copy()
    r0 = cfg.droplet.initial_radius_m
    r1 = cfg.droplet.target_radius_m
    if sp.growth_law == "linear_radius":
        delta = (b - a) * (r1 - r0)
        transformed = edges + delta
    else:
        delta_r2 = (b - a) * (r1 * r1 - r0 * r0)
        transformed = np.sqrt(np.maximum(edges * edges + delta_r2, 1.0e-30))
    return np.maximum.accumulate(np.maximum(transformed, 1.0e-30))


def _radius_grid_at_time(
    base_edges: np.ndarray, model: CloudChargeModel, time_s: float
) -> tuple[np.ndarray, np.ndarray]:
    """Lagrangian sectional radius grid following the prescribed growth law.

    For ``scaled_radius`` the configured radius bounds describe the spectrum
    at the target radius.  During growth the whole lognormal grid is scaled by
    the ratio of the current representative radius to the target radius, so
    the geometric width is preserved while the representative radius grows
    linearly as in the 2012 experiment.
    """
    sp = model.config.spectrum
    d = model.config.droplet
    if sp.growth_law == "scaled_radius":
        if d.growth_time_s <= 0:
            representative = d.target_radius_m
        else:
            fraction = float(np.clip(time_s / d.growth_time_s, 0.0, 1.0))
            representative = d.initial_radius_m + fraction * (d.target_radius_m - d.initial_radius_m)
        moved_edges = base_edges * representative / max(d.target_radius_m, 1.0e-30)
    else:
        moved_edges = _growth_transform(base_edges, model, 0.0, time_s)
    if sp.radius_spacing == "log":
        centers = np.sqrt(moved_edges[:-1] * moved_edges[1:])
    else:
        centers = 0.5 * (moved_edges[:-1] + moved_edges[1:])
    return centers, moved_edges


def _charging_rates(
    model: CloudChargeModel,
    n_pos: np.ndarray,
    n_neg: np.ndarray,
    electric_field: np.ndarray,
    radius_m: np.ndarray,
    charges_e: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    temperature = model.atmosphere.temperature_k
    x = (
        charges_e[None, None, :]
        * ELEMENTARY_CHARGE**2
        / (4.0 * PI * EPSILON_0 * radius_m[None, :, None] * BOLTZMANN * temperature)
    )
    factor_pos = _bernoulli(x)
    factor_neg = _bernoulli(-x)
    drift = np.ones((model.z.size, radius_m.size, 1), dtype=float)
    if model.config.physics.include_drift_charging:
        drift += (
            ELEMENTARY_CHARGE
            * np.abs(electric_field[:, None, None])
            * radius_m[None, :, None]
            / (2.0 * BOLTZMANN * temperature)
        )
    base_pos = 4.0 * PI * radius_m[None, :, None] * model.atmosphere.positive_diffusivity_m2_s
    base_neg = 4.0 * PI * radius_m[None, :, None] * model.atmosphere.negative_diffusivity_m2_s
    rate_pos = base_pos * n_pos[:, None, None] * factor_pos * drift
    rate_neg = base_neg * n_neg[:, None, None] * factor_neg * drift
    # Reflecting boundaries: no transition may leave the represented charge grid.
    rate_pos[:, :, -1] = 0.0
    rate_neg[:, :, 0] = 0.0
    return np.maximum(rate_pos, 0.0), np.maximum(rate_neg, 0.0)


def _advance_charge_spectrum(
    spectrum: np.ndarray,
    rate_pos: np.ndarray,
    rate_neg: np.ndarray,
    dt: float,
    safety: float,
) -> tuple[np.ndarray, int]:
    """Implicit conservative birth-death update in integer charge space.

    Backward Euler produces an M-matrix, so the update is stable and
    non-negative even when strongly attracted ions make the extreme-state
    transition rates large.  A batched Thomas algorithm solves all (z,r)
    tridiagonal systems simultaneously.
    """
    shape = spectrum.shape
    if _HAS_NUMBA:
        solution = _implicit_birth_death_numba(
            np.ascontiguousarray(spectrum, dtype=np.float64),
            np.ascontiguousarray(rate_pos, dtype=np.float64),
            np.ascontiguousarray(rate_neg, dtype=np.float64),
            float(dt),
        )
        out = np.maximum(solution, 0.0)
    else:
        batch = shape[0] * shape[1]
        nq = shape[2]
        rhs = spectrum.reshape(batch, nq)
        rp = rate_pos.reshape(batch, nq)
        rn = rate_neg.reshape(batch, nq)
        main = 1.0 + dt * (rp + rn)
        lower = np.zeros_like(main)
        upper = np.zeros_like(main)
        lower[:, 1:] = -dt * rp[:, :-1]
        upper[:, :-1] = -dt * rn[:, 1:]
        cprime = np.empty_like(main)
        dprime = np.empty_like(rhs)
        denom = np.maximum(main[:, 0], 1.0e-300)
        cprime[:, 0] = upper[:, 0] / denom
        dprime[:, 0] = rhs[:, 0] / denom
        for k in range(1, nq):
            denom = main[:, k] - lower[:, k] * cprime[:, k - 1]
            denom = np.maximum(denom, 1.0e-300)
            cprime[:, k] = upper[:, k] / denom if k < nq - 1 else 0.0
            dprime[:, k] = (rhs[:, k] - lower[:, k] * dprime[:, k - 1]) / denom
        solution = np.empty_like(rhs)
        solution[:, -1] = dprime[:, -1]
        for k in range(nq - 2, -1, -1):
            solution[:, k] = dprime[:, k] - cprime[:, k] * solution[:, k + 1]
        out = np.maximum(solution.reshape(shape), 0.0)
    # Remove roundoff drift in the zeroth moment independently at each z,r.
    old_total = np.sum(spectrum, axis=2)
    new_total = np.sum(out, axis=2)
    factor = np.ones_like(old_total)
    occupied = old_total > 0
    factor[occupied] = old_total[occupied] / np.maximum(new_total[occupied], 1.0e-300)
    out *= factor[:, :, None]
    return out, 1


def _macro_state_from_spectrum(
    macro: np.ndarray, spectrum: np.ndarray, radius_m: np.ndarray, charges_e: np.ndarray, model: CloudChargeModel
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    moments = _moments(spectrum, radius_m, charges_e)
    state = np.zeros((5, model.z.size), dtype=float)
    state[N_POS] = macro[0]
    state[N_NEG] = macro[1]
    state[S_POS] = macro[2]
    state[S_NEG] = macro[3]
    state[P_DROP] = moments["mean_charge_e"] if model.config.spectrum.include_spectrum_feedback else 0.0
    state[P_DROP, model.droplet_number_m3 <= 0] = 0.0
    return state, moments


def _macro_derivative(
    model: CloudChargeModel,
    state: np.ndarray,
    field: FieldSolution,
    rate_pos: np.ndarray,
    rate_neg: np.ndarray,
    spectrum: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    n_pos = state[N_POS]
    n_neg = state[N_NEG]
    s_pos = state[S_POS]
    s_neg = state[S_NEG]
    neutral = np.maximum(model.total_aerosol_m3 - s_pos - s_neg, 0.0)
    electric = field.electric_field_v_m
    dfield = field.dfield_dz_v_m2
    grad_pos = model._gradient(n_pos)
    grad_neg = model._gradient(n_neg)
    divergence_pos = model.atmosphere.positive_mobility_m2_v_s * (n_pos * dfield + electric * grad_pos)
    divergence_neg = model.atmosphere.negative_mobility_m2_v_s * (n_neg * dfield + electric * grad_neg)
    loss_drop_pos = np.sum(rate_pos * spectrum, axis=(1, 2))
    loss_drop_neg = np.sum(rate_neg * spectrum, axis=(1, 2))
    loss_aer_pos = model.beta_pos * neutral * n_pos + model.gamma_pos * s_neg * n_pos
    loss_aer_neg = model.beta_neg * neutral * n_neg + model.gamma_neg * s_pos * n_neg
    recombination = model.config.atmosphere.recombination_m3_s * n_pos * n_neg
    derivative = np.zeros((4, model.z.size), dtype=float)
    derivative[0] = (
        model.config.atmosphere.ion_pair_production_m3_s
        - recombination
        - loss_aer_pos
        - divergence_pos
        - loss_drop_pos
    )
    derivative[1] = (
        model.config.atmosphere.ion_pair_production_m3_s
        - recombination
        - loss_aer_neg
        + divergence_neg
        - loss_drop_neg
    )
    derivative[2] = model.beta_pos * neutral * n_pos - model.gamma_neg * s_pos * n_neg
    derivative[3] = model.beta_neg * neutral * n_neg - model.gamma_pos * s_neg * n_pos
    if model.config.physics.include_ion_diffusion:
        derivative[0] += model.atmosphere.positive_diffusivity_m2_s * _laplacian_no_flux(n_pos, model.dz)
        derivative[1] += model.atmosphere.negative_diffusivity_m2_s * _laplacian_no_flux(n_neg, model.dz)
    return derivative, {
        "droplet_loss_positive_m3_s": loss_drop_pos,
        "droplet_loss_negative_m3_s": loss_drop_neg,
        "aerosol_loss_positive_m3_s": loss_aer_pos,
        "aerosol_loss_negative_m3_s": loss_aer_neg,
        "ion_transport_positive_m3_s": divergence_pos,
        "ion_transport_negative_m3_s": divergence_neg,
    }


def _macro_dt_limit(macro: np.ndarray, derivative: np.ndarray, proposed: float, safety: float) -> float:
    allowed = proposed
    for i in range(4):
        mask = derivative[i] < 0
        if np.any(mask):
            local = -safety * macro[i, mask] / derivative[i, mask]
            allowed = min(allowed, float(np.min(local)))
    return max(min(proposed, allowed), 1.0e-6)


def _enforce_macro(model: CloudChargeModel, macro: np.ndarray, boundary: tuple[float, float, float, float]) -> np.ndarray:
    out = np.asarray(macro, dtype=float).copy()
    out[0] = np.maximum(out[0], 1.0)
    out[1] = np.maximum(out[1], 1.0)
    out[2] = np.maximum(out[2], 0.0)
    out[3] = np.maximum(out[3], 0.0)
    charged = out[2] + out[3]
    limit = 0.999999 * model.total_aerosol_m3
    over = charged > limit
    if np.any(over):
        scale = limit[over] / charged[over]
        out[2, over] *= scale
        out[3, over] *= scale
    npos_clear, nneg_clear, spos_clear, sneg_clear = boundary
    # Positive-ion number flux enters from the top; negative-ion number flux enters from the bottom.
    out[0, 0] = npos_clear
    out[1, -1] = nneg_clear
    out[2, 0] = spos_clear
    out[3, -1] = sneg_clear
    if not np.all(np.isfinite(out)):
        raise FloatingPointError("Non-finite macro state in spectral solver.")
    return out


def _normalise_spectrum(spectrum: np.ndarray, target_number: np.ndarray) -> np.ndarray:
    total = np.sum(spectrum, axis=(1, 2))
    cloudy = target_number > 0
    factor = np.ones_like(total)
    factor[cloudy] = target_number[cloudy] / np.maximum(total[cloudy], 1.0e-300)
    spectrum = spectrum * factor[:, None, None]
    spectrum[~cloudy] = 0.0
    return spectrum


def _snapshot(
    model: CloudChargeModel,
    macro: np.ndarray,
    spectrum: np.ndarray,
    radius_m: np.ndarray,
    radius_edges_m: np.ndarray,
    charges_e: np.ndarray,
    time_s: float,
    field: FieldSolution,
) -> dict[str, np.ndarray | float]:
    _, moments = _macro_state_from_spectrum(macro, spectrum, radius_m, charges_e, model)
    neutral_aerosol = np.maximum(model.total_aerosol_m3 - macro[2] - macro[3], 0.0)
    snap: dict[str, np.ndarray | float] = {
        "time_s": float(time_s),
        "n_positive_m3": macro[0].copy(),
        "n_negative_m3": macro[1].copy(),
        "s_positive_m3": macro[2].copy(),
        "s_negative_m3": macro[3].copy(),
        "neutral_aerosol_m3": neutral_aerosol,
        "droplet_number_m3": moments["number_m3"].copy(),
        "droplet_charge_e": moments["mean_charge_e"].copy(),
        "droplet_charge_std_e": moments["charge_std_e"].copy(),
        "droplet_mean_radius_m": moments["mean_radius_m"].copy(),
        "droplet_radius_std_m": moments["radius_std_m"].copy(),
        "droplet_positive_fraction": moments["positive_fraction"].copy(),
        "droplet_negative_fraction": moments["negative_fraction"].copy(),
        "droplet_neutral_fraction": moments["neutral_fraction"].copy(),
        "droplet_modal_charge_e": moments["modal_charge_e"].copy(),
        "droplet_radius_charge_covariance_m_e": moments["radius_charge_covariance_m_e"].copy(),
        "droplet_size_spectrum_m3": moments["size_number_m3"].copy(),
        "droplet_charge_spectrum_m3": moments["charge_number_m3"].copy(),
        "radius_centers_m": radius_m.copy(),
        "radius_edges_m": radius_edges_m.copy(),
        "charge_states_e": charges_e.copy(),
        "electric_field_v_m": field.electric_field_v_m.copy(),
        "dfield_dz_v_m2": field.dfield_dz_v_m2.copy(),
        "charge_density_c_m3": field.charge_density_c_m3.copy(),
        "current_density_a_m2": field.current_density_a_m2.copy(),
    }
    if model.config.spectrum.store_full_spectrum:
        snap["droplet_spectrum_m3"] = spectrum.astype(np.float32, copy=True)
    return snap


def _fit_tau(time: np.ndarray, value: np.ndarray, start: float, end: float) -> float:
    mask = (time >= start) & (time <= end)
    t = time[mask]
    y = value[mask]
    if t.size < 6:
        return float("nan")
    y0 = float(y[0])
    yinf = float(np.mean(y[-max(3, y.size // 10) :]))
    delta = yinf - y0
    if abs(delta) < 1.0e-8:
        return float("nan")
    remaining = (yinf - y) / delta
    good = (remaining > 0.05) & (remaining < 0.95)
    if np.count_nonzero(good) < 3:
        return float("nan")
    slope, _ = np.polyfit(t[good] - start, np.log(remaining[good]), 1)
    return float(-1.0 / slope) if slope < 0 else float("nan")


def run_paper2012_spectrum(
    model: CloudChargeModel,
    progress_callback: Optional["ProgressCallback"] = None,
    cancel_callback: Optional["CancelCallback"] = None,
) -> SimulationResult:
    """Self-consistent size-charge spectral extension of Zhou & Tinsley (2012).

    The continuous distribution N(r,Q,z,t) is represented as bin-integrated
    N[i,k,j,t], where k is an integer number of elementary charges.  Ion
    attachment is a conservative birth-death process in charge space.  The
    first charge moment feeds back into Gauss's law, while the full spectrum is
    used for positive and negative ion loss rates.
    """

    cfg = model.config
    cfg.validate()
    td = cfg.time_dependent
    duration = float(td.total_duration_s)
    step_time = float(td.step_time_s)
    base_radius_m, base_radius_edges_m = _radius_grid(model)
    radius_m, radius_edges_m = _radius_grid_at_time(base_radius_edges_m, model, 0.0)
    charges_e = np.arange(cfg.spectrum.charge_min_e, cfg.spectrum.charge_max_e + 1, dtype=float)
    size_weights = _size_bin_weights(model, radius_edges_m)
    charge_weights = _charge_bin_weights(model, charges_e)
    spectrum = (
        model.droplet_number_m3[:, None, None]
        * size_weights[None, :, None]
        * charge_weights[None, None, :]
    )

    clear = model._homogeneous_equilibrium_corrected()
    macro = np.empty((4, model.z.size), dtype=float)
    for i, value in enumerate(clear):
        macro[i] = value
    macro = _enforce_macro(model, macro, clear)

    constrained = cfg.spectrum.carrier_coupling == "paper_constrained"
    eq1_macro: np.ndarray | None = None
    eq2_macro: np.ndarray | None = None
    tau_drop, tau_source = _estimate_tau(cfg)
    tau_aerosol = 0.81 * tau_drop
    tau_ion = float(np.clip(0.125 * tau_drop, 20.0, 150.0))
    if constrained:
        if progress_callback:
            progress_callback(0.01, "计算谱模式第一阶段的2007稳态载流子目标")
        eq1_result = _equilibrium_result(cfg, td.initial_current_density_a_m2, td.initial_ion_pair_production_m3_s)
        if cancel_callback and cancel_callback():
            raise SimulationCancelled("Simulation cancelled by user.")
        if progress_callback:
            progress_callback(0.03, "计算Jz/q阶跃后的稳态载流子目标")
        eq2_result = _equilibrium_result(cfg, td.second_current_density_a_m2, td.second_ion_pair_production_m3_s)
        e1 = eq1_result.snapshots[float(eq1_result.snapshot_times_s[-1])]
        e2 = eq2_result.snapshots[float(eq2_result.snapshot_times_s[-1])]
        eq1_macro = np.vstack([e1["n_positive_m3"], e1["n_negative_m3"], e1["s_positive_m3"], e1["s_negative_m3"]]).astype(float)
        eq2_macro = np.vstack([e2["n_positive_m3"], e2["n_negative_m3"], e2["s_positive_m3"], e2["s_negative_m3"]]).astype(float)

    z_gradient = np.gradient(model.droplet_number_m3, model.z, edge_order=2)
    cloudy = model.droplet_number_m3 > 0.01 * model.droplet_peak_m3
    top_candidates = np.where(cloudy & (model.z <= cfg.domain.cloud_start_m + cfg.domain.edge_ramp_m + 1.0e-9))[0]
    bottom_start = cfg.domain.cloud_start_m + cfg.domain.edge_ramp_m + cfg.domain.core_width_m
    bottom_candidates = np.where(cloudy & (model.z >= bottom_start - 1.0e-9))[0]
    top_index = int(top_candidates[np.argmax(z_gradient[top_candidates])])
    bottom_index = int(bottom_candidates[np.argmin(z_gradient[bottom_candidates])])
    core_index = int(np.argmin(np.abs(model.z - (cfg.domain.cloud_start_m + cfg.domain.edge_ramp_m + 0.5 * cfg.domain.core_width_m))))

    requested = sorted(
        set(
            [0.0, duration, step_time, cfg.droplet.growth_time_s]
            + [min(duration, max(0.0, float(x))) for x in cfg.numerics.snapshot_times_s]
        )
    )
    snapshots: dict[float, dict[str, np.ndarray | float]] = {}
    next_snapshot = 0

    diag_time: list[float] = []
    top_mean: list[float] = []
    base_mean: list[float] = []
    top_std: list[float] = []
    base_std: list[float] = []
    top_positive: list[float] = []
    base_negative: list[float] = []
    top_radius: list[float] = []
    base_radius: list[float] = []
    top_mode: list[float] = []
    base_mode: list[float] = []
    current_residual: list[float] = []
    number_error: list[float] = []
    charge_edge_fraction: list[float] = []
    radius_edge_fraction: list[float] = []
    forcing_j: list[float] = []
    forcing_q: list[float] = []
    max_subcycles = 1
    step_count = 0
    min_dt = float("inf")
    max_dt = 0.0
    time_s = 0.0
    next_diag = 0.0

    def carrier_target(t: float) -> np.ndarray:
        if not constrained or eq1_macro is None or eq2_macro is None:
            return macro
        if t < step_time:
            if cfg.droplet.growth_time_s > 0 and t < cfg.droplet.growth_time_s:
                frac = float(np.clip(t / cfg.droplet.growth_time_s, 0.0, 1.0))
                initial = np.empty_like(eq1_macro)
                for ii, value in enumerate(clear):
                    initial[ii] = value
                return initial + frac * (eq1_macro - initial)
            return eq1_macro
        return eq2_macro

    def set_forcing(t: float) -> tuple[float, float]:
        if t < step_time:
            j = td.initial_current_density_a_m2
            ion_rate = td.initial_ion_pair_production_m3_s
        else:
            j = td.second_current_density_a_m2
            ion_rate = td.second_ion_pair_production_m3_s
        cfg.atmosphere.current_density_a_m2 = j
        cfg.atmosphere.ion_pair_production_m3_s = ion_rate
        return j, ion_rate

    def field_from_current() -> tuple[np.ndarray, dict[str, np.ndarray], FieldSolution]:
        nonlocal macro
        state, moments = _macro_state_from_spectrum(macro, spectrum, radius_m, charges_e, model)
        if constrained:
            # In the paper-constrained spectral mode, Eq. (10)/(19), Jz=sigma E,
            # is the algebraic field constraint.  The evolving spectrum supplies
            # the actual charge density; its mismatch with eps0*dE/dz is reported
            # as the transient Poisson residual rather than being removed by an
            # artificial equal increment to n+ and n-.
            sigma_number = np.maximum(
                model.atmosphere.positive_mobility_m2_v_s * state[N_POS]
                + model.atmosphere.negative_mobility_m2_v_s * state[N_NEG],
                1.0e-30,
            )
            electric = cfg.atmosphere.current_density_a_m2 / (ELEMENTARY_CHARGE * sigma_number)
            dfield = np.gradient(electric, model.z, edge_order=2)
            rho = ELEMENTARY_CHARGE * (
                state[N_POS] - state[N_NEG] + state[S_POS] - state[S_NEG]
                + state[P_DROP] * model.droplet_number_m3
            )
            current = ELEMENTARY_CHARGE * sigma_number * electric
            field = FieldSolution(
                state=state.copy(),
                electric_field_v_m=electric,
                dfield_dz_v_m2=dfield,
                charge_density_c_m3=rho,
                current_density_a_m2=current,
                projection_iterations=0,
                relative_current_residual=float(
                    np.max(np.abs(current - cfg.atmosphere.current_density_a_m2))
                    / max(abs(cfg.atmosphere.current_density_a_m2), 1.0e-30)
                ),
            )
            return state, moments, field
        field = model.solve_field(state)
        macro[0] = field.state[N_POS]
        macro[1] = field.state[N_NEG]
        macro[2] = field.state[S_POS]
        macro[3] = field.state[S_NEG]
        return field.state, moments, field

    def record(t: float, field: FieldSolution | None = None) -> FieldSolution:
        j, ion_rate = set_forcing(t + (1.0e-9 if t >= step_time else 0.0))
        state, moments, local_field = field_from_current() if field is None else (*_macro_state_from_spectrum(macro, spectrum, radius_m, charges_e, model), field)
        # The tuple expansion above yields state, moments, field for either branch.
        total = moments["number_m3"]
        target = model.droplet_number_m3
        mask = target > 0
        err = float(np.max(np.abs(total[mask] - target[mask]) / np.maximum(target[mask], 1.0))) if np.any(mask) else 0.0
        all_number = float(np.sum(spectrum))
        edge_q = float(np.sum(spectrum[:, :, 0]) + np.sum(spectrum[:, :, -1])) / max(all_number, 1.0e-300)
        edge_r = float(np.sum(spectrum[:, 0, :]) + np.sum(spectrum[:, -1, :])) / max(all_number, 1.0e-300)
        diag_time.append(float(t))
        top_mean.append(float(moments["mean_charge_e"][top_index]))
        base_mean.append(float(moments["mean_charge_e"][bottom_index]))
        top_std.append(float(moments["charge_std_e"][top_index]))
        base_std.append(float(moments["charge_std_e"][bottom_index]))
        top_positive.append(float(moments["positive_fraction"][top_index]))
        base_negative.append(float(moments["negative_fraction"][bottom_index]))
        top_radius.append(float(moments["mean_radius_m"][top_index]))
        base_radius.append(float(moments["mean_radius_m"][bottom_index]))
        top_mode.append(float(moments["modal_charge_e"][top_index]))
        base_mode.append(float(moments["modal_charge_e"][bottom_index]))
        current_residual.append(float(local_field.relative_current_residual))
        number_error.append(err)
        charge_edge_fraction.append(edge_q)
        radius_edge_fraction.append(edge_r)
        forcing_j.append(j)
        forcing_q.append(ion_rate)
        return local_field

    set_forcing(0.0)
    state, moments, field = field_from_current()
    snapshots[0.0] = _snapshot(model, macro, spectrum, radius_m, radius_edges_m, charges_e, 0.0, field)
    next_snapshot = 1 if requested and requested[0] == 0.0 else 0
    record(0.0, field)
    next_diag = cfg.numerics.diagnostics_interval_s
    if progress_callback:
        progress_callback(0.0, "粒径—电荷谱初始化完成，开始N(r,k,z,t)积分")

    while time_s < duration - 1.0e-12:
        if cancel_callback and cancel_callback():
            raise SimulationCancelled("Simulation cancelled by user.")
        j_now, q_now = set_forcing(time_s)
        state, moments, field = field_from_current()
        rate_pos, rate_neg = _charging_rates(
            model, macro[0], macro[1], field.electric_field_v_m, radius_m, charges_e
        )
        derivative, _ = _macro_derivative(model, state, field, rate_pos, rate_neg, spectrum)
        proposed = min(cfg.numerics.max_dt_s, duration - time_s)
        future_events = [x for x in requested + [next_diag] if x > time_s + 1.0e-10]
        if future_events:
            proposed = min(proposed, min(future_events) - time_s)
        dt = proposed if constrained else _macro_dt_limit(macro, derivative, proposed, cfg.numerics.positivity_safety)
        if dt <= 0:
            raise RuntimeError("Non-positive time step in spectral solver.")

        # Lagrangian radius sections follow the prescribed growth law without
        # numerically diffusing the size distribution between fixed bins.
        radius_m, radius_edges_m = _radius_grid_at_time(
            base_radius_edges_m, model, time_s + 0.5 * dt
        )
        rate_pos, rate_neg = _charging_rates(
            model, macro[0], macro[1], field.electric_field_v_m, radius_m, charges_e
        )
        spectrum, subcycles = _advance_charge_spectrum(
            spectrum, rate_pos, rate_neg, dt, cfg.spectrum.charge_subcycle_safety
        )
        max_subcycles = max(max_subcycles, subcycles)

        if constrained:
            target = carrier_target(time_s + 0.5 * dt)
            macro[:2] = target[:2] + (macro[:2] - target[:2]) * math.exp(-dt / tau_ion)
            macro[2:] = target[2:] + (macro[2:] - target[2:]) * math.exp(-dt / tau_aerosol)
            macro = _enforce_macro(model, macro, clear)
        else:
            # Experimental fully direct coupling: full spectrum supplies W+ and W-.
            derivative, _ = _macro_derivative(model, state, field, rate_pos, rate_neg, spectrum)
            macro = _enforce_macro(model, macro + dt * derivative, clear)
        if cfg.spectrum.normalize_number_each_step:
            spectrum = _normalise_spectrum(spectrum, model.droplet_number_m3)

        time_s += dt
        radius_m, radius_edges_m = _radius_grid_at_time(
            base_radius_edges_m, model, time_s
        )
        step_count += 1
        min_dt = min(min_dt, dt)
        max_dt = max(max_dt, dt)

        while next_snapshot < len(requested) and time_s >= requested[next_snapshot] - 1.0e-8:
            snap_time = float(requested[next_snapshot])
            set_forcing(snap_time + (1.0e-9 if snap_time >= step_time else 0.0))
            _, _, snap_field = field_from_current()
            snapshots[snap_time] = _snapshot(
                model, macro, spectrum, radius_m, radius_edges_m, charges_e, snap_time, snap_field
            )
            next_snapshot += 1

        if time_s >= next_diag - 1.0e-8:
            set_forcing(time_s + (1.0e-9 if time_s >= step_time else 0.0))
            record(time_s)
            next_diag += cfg.numerics.diagnostics_interval_s

        if progress_callback and (step_count == 1 or step_count % 25 == 0):
            mm = _moments(spectrum, radius_m, charges_e)
            progress_callback(
                min(1.0, time_s / duration),
                f"谱积分 t={time_s:.0f}s, Jz={j_now/1e-12:.2f} pA m⁻², "
                f"top/base <k>={mm['mean_charge_e'][top_index]:+.1f}/{mm['mean_charge_e'][bottom_index]:+.1f} e",
            )

    set_forcing(duration + 1.0e-9)
    final_state, final_moments, final_field = field_from_current()
    snapshots[duration] = _snapshot(
        model, macro, spectrum, radius_m, radius_edges_m, charges_e, duration, final_field
    )
    if not diag_time or abs(diag_time[-1] - duration) > 1.0e-6:
        record(duration, final_field)

    tarr = np.asarray(diag_time, dtype=float)
    top_arr = np.asarray(top_mean, dtype=float)
    base_arr = np.asarray(base_mean, dtype=float)
    tau_top = _fit_tau(tarr, top_arr, step_time, duration)
    tau_base = _fit_tau(tarr, base_arr, step_time, duration)
    final_rho = final_field.charge_density_c_m3
    top_mask = (model.z >= cfg.domain.cloud_start_m) & (
        model.z <= cfg.domain.cloud_start_m + cfg.domain.edge_ramp_m
    )
    base_mask = (model.z >= bottom_start) & (
        model.z <= bottom_start + cfg.domain.edge_ramp_m
    )
    upper_column = float(np.trapezoid(final_rho[top_mask], model.z[top_mask]))
    lower_column = float(np.trapezoid(final_rho[base_mask], model.z[base_mask]))
    poisson_residual = np.gradient(final_field.electric_field_v_m, model.z, edge_order=2) - final_rho / EPSILON_0

    diagnostics: dict[str, np.ndarray | float | str | bool] = {
        "solver_mode": "paper2012_spectrum",
        "time_s": tarr,
        "top_droplet_charge_e": top_arr,
        "bottom_droplet_charge_e": base_arr,
        "top_droplet_charge_std_e": np.asarray(top_std),
        "bottom_droplet_charge_std_e": np.asarray(base_std),
        "top_positive_fraction": np.asarray(top_positive),
        "bottom_negative_fraction": np.asarray(base_negative),
        "top_mean_radius_m": np.asarray(top_radius),
        "bottom_mean_radius_m": np.asarray(base_radius),
        "top_modal_charge_e": np.asarray(top_mode),
        "bottom_modal_charge_e": np.asarray(base_mode),
        "scheduled_current_density_a_m2": np.asarray(forcing_j),
        "scheduled_ion_pair_production_m3_s": np.asarray(forcing_q),
        "relative_current_residual": np.asarray(current_residual),
        "droplet_number_conservation_error": np.asarray(number_error),
        "charge_grid_edge_fraction": np.asarray(charge_edge_fraction),
        "radius_grid_edge_fraction": np.asarray(radius_edge_fraction),
        "post_step_tau_top_s": tau_top,
        "post_step_tau_bottom_s": tau_base,
        "carrier_response_tau_ion_s": tau_ion,
        "carrier_response_tau_aerosol_s": tau_aerosol,
        "paper_droplet_tau_reference_s": tau_drop,
        "paper_droplet_tau_source": tau_source,
        "spectral_carrier_coupling": cfg.spectrum.carrier_coupling,
        "upper_peak_droplet_charge_e": float(np.max(final_moments["mean_charge_e"][top_mask])),
        "lower_peak_droplet_charge_e": float(np.min(final_moments["mean_charge_e"][base_mask])),
        "upper_integrated_space_charge_c_m2": upper_column,
        "lower_integrated_space_charge_c_m2": lower_column,
        "opposite_droplet_sign_test": bool(
            np.max(final_moments["mean_charge_e"][top_mask]) > 0
            and np.min(final_moments["mean_charge_e"][base_mask]) < 0
        ),
        "opposite_space_charge_sign_test": bool(upper_column > 0 and lower_column < 0),
        "top_spectrum_index": float(top_index),
        "bottom_spectrum_index": float(bottom_index),
        "core_spectrum_index": float(core_index),
        "time_steps": float(step_count),
        "min_dt_s": float(min_dt if math.isfinite(min_dt) else 0.0),
        "max_dt_s": float(max_dt),
        "max_charge_subcycles": float(max_subcycles),
        "final_max_relative_current_residual": float(final_field.relative_current_residual),
        "final_poisson_residual_v_m2": poisson_residual,
        "final_max_number_conservation_error": float(number_error[-1]),
        "final_charge_grid_edge_fraction": float(charge_edge_fraction[-1]),
        "final_radius_grid_edge_fraction": float(radius_edge_fraction[-1]),
    }
    metadata: dict[str, object] = {
        "config": cfg.to_dict(),
        "atmosphere": asdict(model.atmosphere),
        "attachment": {
            "beta_positive_m3_s": model.beta_pos,
            "beta_negative_m3_s": model.beta_neg,
            "gamma_positive_m3_s": model.gamma_pos,
            "gamma_negative_m3_s": model.gamma_neg,
            **model.attachment.metadata,
        },
        "spectral_definition": {
            "state": "N(r_i, k, z_j, t) bin-integrated number concentration [m^-3]",
            "charge_variable": "k = integer elementary charges per droplet; q is reserved for ion-pair production",
            "radius_bins": int(radius_m.size),
            "charge_states": int(charges_e.size),
            "spectral_cells": int(spectrum.size),
            "charge_process": "conservative ion-attachment birth-death master equation",
            "radius_process": cfg.spectrum.growth_law,
            "feedback": "first charge moment enters Gauss law; full spectrum enters W+ and W-",
            "carrier_coupling": cfg.spectrum.carrier_coupling,
        },
        "model_limits": (
            "One-dimensional prescribed cloud-droplet number profile. The spectral model includes prescribed condensational "
            "growth and ion diffusion/drift charging, but not droplet collision-coalescence, sedimentation, turbulent mixing, "
            "evaporation residues, or charge-dependent breakup."
        ),
    }
    if progress_callback:
        progress_callback(1.0, "粒径—电荷二维谱模拟完成")
    return SimulationResult(
        z_m=model.z.copy(),
        snapshot_times_s=np.asarray(sorted(snapshots.keys()), dtype=float),
        snapshots=snapshots,
        diagnostics=diagnostics,
        metadata=metadata,
    )
