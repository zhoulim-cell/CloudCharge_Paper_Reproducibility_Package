from __future__ import annotations

"""Steady boundary-value solver reproducing Zhou and Tinsley (2007).

The paper uses a downward-positive coordinate z and downward-positive fair-weather
current density Jz. Positive ions enter through the upper boundary and negative
ions enter through the lower boundary. This mixed inflow boundary condition is
essential: imposing both ion concentrations at the same boundary selects an
unphysical branch and can give the same droplet-charge sign at both cloud edges.
"""

from dataclasses import asdict
import math
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np
from scipy.integrate import solve_bvp
from scipy.optimize import least_squares

from .constants import BOLTZMANN, ELEMENTARY_CHARGE, EPSILON_0, PI
from .solver import SimulationCancelled, SimulationResult

if TYPE_CHECKING:  # pragma: no cover
    from .solver import CloudChargeModel, ProgressCallback, CancelCallback


def _stable_xy_log_ratio_over_difference(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return x*y*ln(x/y)/(x-y), with the x -> y limit evaluated stably."""
    x = np.maximum(np.asarray(x, dtype=float), 1.0e-300)
    y = np.maximum(np.asarray(y, dtype=float), 1.0e-300)
    ratio = x / y
    h = ratio - 1.0
    out = np.empty_like(ratio)
    close = np.abs(h) < 1.0e-5
    hc = h[close]
    # y*(1+h)*log1p(h)/h; series through fourth order.
    out[close] = y[close] * (1.0 + hc) * (
        1.0 - hc / 2.0 + hc**2 / 3.0 - hc**3 / 4.0 + hc**4 / 5.0
    )
    out[~close] = x[~close] * y[~close] * np.log(ratio[~close]) / (
        x[~close] - y[~close]
    )
    return out


def _homogeneous_clear_air_equilibrium(model: "CloudChargeModel") -> tuple[float, float, float, float, float]:
    """Clear-air equilibrium satisfying ion balance and zero total space charge."""
    cfg = model.config
    q = cfg.atmosphere.ion_pair_production_m3_s
    alpha = cfg.atmosphere.recombination_m3_s
    st = cfg.aerosol.total_number_m3
    scale = math.sqrt(q / alpha)

    def aerosol_species(n_pos: float, n_neg: float) -> tuple[float, float, float]:
        ratio_pos = model.beta_pos * n_pos / max(model.gamma_neg * n_neg, 1.0e-300)
        ratio_neg = model.beta_neg * n_neg / max(model.gamma_pos * n_pos, 1.0e-300)
        neutral = st / (1.0 + ratio_pos + ratio_neg)
        return neutral, neutral * ratio_pos, neutral * ratio_neg

    def residual(log_n: np.ndarray) -> np.ndarray:
        n_pos, n_neg = np.exp(log_n)
        neutral, s_pos, s_neg = aerosol_species(float(n_pos), float(n_neg))
        ion_balance = q - alpha * n_pos * n_neg - neutral * (
            model.beta_pos * n_pos + model.beta_neg * n_neg
        )
        charge_balance = n_pos - n_neg + s_pos - s_neg
        return np.array([ion_balance / q, charge_balance / scale])

    solution = least_squares(
        residual,
        np.log([0.7 * scale, 0.7 * scale]),
        xtol=1.0e-13,
        ftol=1.0e-13,
        gtol=1.0e-13,
        max_nfev=2000,
    )
    if not solution.success:
        raise RuntimeError(f"Clear-air equilibrium solve failed: {solution.message}")
    n_pos, n_neg = np.exp(solution.x)
    neutral, s_pos, s_neg = aerosol_species(float(n_pos), float(n_neg))
    return float(n_pos), float(n_neg), float(neutral), float(s_pos), float(s_neg)


def _boundary_masks(model: "CloudChargeModel") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = model.config.domain
    top_start = d.cloud_start_m
    top_end = top_start + d.edge_ramp_m
    core_end = top_end + d.core_width_m
    cloud_end = core_end + d.edge_ramp_m
    z = model.z
    upper = (z >= top_start) & (z <= top_end)
    core = (z > top_end) & (z < core_end)
    lower = (z >= core_end) & (z <= cloud_end)
    return upper, core, lower


def run_paper2007_equilibrium(
    model: "CloudChargeModel",
    progress_callback: Optional["ProgressCallback"] = None,
    cancel_callback: Optional["CancelCallback"] = None,
) -> SimulationResult:
    """Solve equations (10), (16), (19), (23), and (25) as a BVP.

    The two boundary conditions are imposed at the *inflow* sides of the ion
    transport: n+ at cloud top and n- at cloud base for downward Jz. This is
    the physically correct mixed boundary condition for the fair-weather
    global electric circuit and is what produces positive cloud-top droplets
    and negative cloud-base droplets.
    """
    cfg = model.config
    jz = cfg.atmosphere.current_density_a_m2
    if jz <= 1.0e-18:
        raise ValueError("paper2007_bvp mode requires the fair-weather convention Jz > 0 downward.")
    if cancel_callback and cancel_callback():
        raise SimulationCancelled("Simulation cancelled by user.")
    if progress_callback:
        progress_callback(0.03, "构造2007论文稳态边值问题")

    z = model.z
    nd = model.droplet_number_m3
    radius = cfg.droplet.target_radius_m
    st = cfg.aerosol.total_number_m3
    q = cfg.atmosphere.ion_pair_production_m3_s
    alpha = cfg.atmosphere.recombination_m3_s
    mu_pos = model.atmosphere.positive_mobility_m2_v_s
    mu_neg = model.atmosphere.negative_mobility_m2_v_s
    d_pos = model.atmosphere.positive_diffusivity_m2_s
    d_neg = model.atmosphere.negative_diffusivity_m2_s
    temperature = model.atmosphere.temperature_k
    charge_factor = 4.0 * PI * EPSILON_0 * BOLTZMANN * temperature * radius / (
        ELEMENTARY_CHARGE**2
    )

    n_pos_clear, n_neg_clear, neutral_clear, s_pos_clear, s_neg_clear = (
        _homogeneous_clear_air_equilibrium(model)
    )

    def droplet_number_at(points: np.ndarray) -> np.ndarray:
        return np.interp(points, z, nd)

    def aerosol_species(n_pos: np.ndarray, n_neg: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ratio_pos = model.beta_pos * n_pos / np.maximum(model.gamma_neg * n_neg, 1.0e-300)
        ratio_neg = model.beta_neg * n_neg / np.maximum(model.gamma_pos * n_pos, 1.0e-300)
        neutral = st / (1.0 + ratio_pos + ratio_neg)
        return neutral, neutral * ratio_pos, neutral * ratio_neg

    def equations(points: np.ndarray, log_n: np.ndarray) -> np.ndarray:
        n_pos = np.exp(np.clip(log_n[0], -100.0, 100.0))
        n_neg = np.exp(np.clip(log_n[1], -100.0, 100.0))
        local_nd = droplet_number_at(points)
        neutral, s_pos, s_neg = aerosol_species(n_pos, n_neg)

        p_drop = charge_factor * np.log(
            np.maximum(d_pos * n_pos, 1.0e-300)
            / np.maximum(d_neg * n_neg, 1.0e-300)
        )
        charge_number = n_pos - n_neg + s_pos - s_neg + p_drop * local_nd
        dfield_dz = ELEMENTARY_CHARGE * charge_number / EPSILON_0
        conductivity_number = np.maximum(mu_pos * n_pos + mu_neg * n_neg, 1.0e-300)
        electric_field = jz / (ELEMENTARY_CHARGE * conductivity_number)

        droplet_loss = (
            4.0
            * PI
            * radius
            * local_nd
            * _stable_xy_log_ratio_over_difference(d_pos * n_pos, d_neg * n_neg)
        )
        common_balance = q - alpha * n_pos * n_neg - neutral * (
            model.beta_pos * n_pos + model.beta_neg * n_neg
        ) - droplet_loss

        dn_pos_dz = (
            common_balance - mu_pos * n_pos * dfield_dz
        ) / np.maximum(electric_field * mu_pos, 1.0e-300)
        # d[Jz]/dz=0 supplies the second ODE exactly.
        dn_neg_dz = (
            -(dfield_dz / electric_field) * conductivity_number - mu_pos * dn_pos_dz
        ) / mu_neg
        return np.vstack((dn_pos_dz / n_pos, dn_neg_dz / n_neg))

    def boundary_condition(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return np.array(
            [left[0] - math.log(n_pos_clear), right[1] - math.log(n_neg_clear)]
        )

    # Local scalar ion-balance estimate supplies a smooth, physics-based initial guess.
    neutral_guess = neutral_clear
    aerosol_frequency = neutral_guess * 0.5 * (model.beta_pos + model.beta_neg)
    droplet_frequency = 4.0 * PI * radius * nd * 0.5 * (d_pos + d_neg)
    total_frequency = aerosol_frequency + droplet_frequency
    n_scalar = (
        -total_frequency + np.sqrt(total_frequency**2 + 4.0 * alpha * q)
    ) / (2.0 * alpha)
    n_scalar = np.maximum(n_scalar, 1.0e5)
    n_pos_guess = n_scalar * n_pos_clear / math.sqrt(n_pos_clear * n_neg_clear)
    n_neg_guess = n_scalar * n_neg_clear / math.sqrt(n_pos_clear * n_neg_clear)
    initial_guess = np.vstack((np.log(n_pos_guess), np.log(n_neg_guess)))

    if progress_callback:
        progress_callback(0.12, "求解正离子上边界/负离子下边界混合边值条件")
    solution = solve_bvp(
        equations,
        boundary_condition,
        z,
        initial_guess,
        tol=cfg.numerics.bvp_tolerance,
        max_nodes=cfg.numerics.bvp_max_nodes,
        verbose=0,
    )
    if cancel_callback and cancel_callback():
        raise SimulationCancelled("Simulation cancelled by user.")
    if not solution.success:
        raise RuntimeError(
            "2007 paper BVP solver did not converge: "
            f"{solution.message}; nodes={solution.x.size}, "
            f"max residual={float(np.max(solution.rms_residuals)):.3e}"
        )
    if progress_callback:
        progress_callback(0.80, "边值问题收敛，计算电荷分配与观测检验")

    log_solution = solution.sol(z)
    n_pos = np.exp(log_solution[0])
    n_neg = np.exp(log_solution[1])
    neutral, s_pos, s_neg = aerosol_species(n_pos, n_neg)
    p_unmasked = charge_factor * np.log(
        np.maximum(d_pos * n_pos, 1.0e-300)
        / np.maximum(d_neg * n_neg, 1.0e-300)
    )
    p_drop = p_unmasked.copy()
    p_drop[nd <= max(1.0, 1.0e-12 * float(np.max(nd)))] = 0.0
    charge_density = ELEMENTARY_CHARGE * (
        n_pos - n_neg + s_pos - s_neg + p_unmasked * nd
    )
    dfield_dz = charge_density / EPSILON_0
    conductivity_number = mu_pos * n_pos + mu_neg * n_neg
    electric_field = jz / (ELEMENTARY_CHARGE * conductivity_number)
    current = ELEMENTARY_CHARGE * conductivity_number * electric_field

    numeric_dfield = np.gradient(electric_field, z, edge_order=2)
    poisson_residual = numeric_dfield - dfield_dz
    current_residual = float(
        np.max(np.abs(current - jz)) / max(abs(jz), 1.0e-30)
    )

    upper, core, lower = _boundary_masks(model)
    top_peak = float(np.max(p_drop[upper]))
    bottom_peak = float(np.min(p_drop[lower]))
    top_surface_charge = float(np.trapezoid(charge_density[upper], z[upper]))
    bottom_surface_charge = float(np.trapezoid(charge_density[lower], z[lower]))
    top_rho_peak = float(np.max(charge_density[upper]))
    bottom_rho_peak = float(np.min(charge_density[lower]))
    upper_cloudy = upper & (nd > 0.01 * float(np.max(nd)))
    lower_cloudy = lower & (nd > 0.01 * float(np.max(nd)))
    upper_positive_fraction = float(np.mean(p_drop[upper_cloudy] > 0.0))
    lower_negative_fraction = float(np.mean(p_drop[lower_cloudy] < 0.0))
    outside_mask = model.cloud_shape < 1.0e-8
    core_mask = core & (model.cloud_shape > 0.999)
    outside_field = float(np.mean(electric_field[outside_mask]))
    inside_field = float(np.mean(electric_field[core_mask])) if np.any(core_mask) else float(np.max(electric_field))

    sign_pass = bool(top_peak > 0.0 and bottom_peak < 0.0)
    space_charge_sign_pass = bool(top_surface_charge > 0.0 and bottom_surface_charge < 0.0)
    conductivity_step_pass = bool(abs(inside_field) > abs(outside_field))
    observed_top_e = 85.0
    observed_bottom_e = -65.0

    snapshot = {
        "time_s": float(cfg.numerics.duration_s),
        "n_positive_m3": n_pos.copy(),
        "n_negative_m3": n_neg.copy(),
        "s_positive_m3": s_pos.copy(),
        "s_negative_m3": s_neg.copy(),
        "neutral_aerosol_m3": neutral.copy(),
        "droplet_charge_e": p_drop.copy(),
        "droplet_number_m3": nd.copy(),
        "droplet_radius_m": float(radius),
        "electric_field_v_m": electric_field.copy(),
        "dfield_dz_v_m2": dfield_dz.copy(),
        "charge_density_c_m3": charge_density.copy(),
        "current_density_a_m2": current.copy(),
    }
    snapshots = {
        0.0: {**snapshot, "time_s": 0.0},
        float(cfg.numerics.duration_s): snapshot,
    }

    diagnostics: dict[str, np.ndarray | float | str | bool] = {
        "solver_mode": "paper2007_bvp",
        "time_s": np.array([0.0, cfg.numerics.duration_s]),
        "max_abs_electric_field_v_m": np.array([np.max(np.abs(electric_field))] * 2),
        "max_abs_droplet_charge_e": np.array([np.max(np.abs(p_drop))] * 2),
        "relative_current_residual": np.array([current_residual] * 2),
        "column_net_charge_c_m2": np.array([np.trapezoid(charge_density, z)] * 2),
        "final_poisson_residual_v_m2": poisson_residual,
        "final_max_relative_current_residual": current_residual,
        "time_steps": 0.0,
        "min_dt_s": 0.0,
        "max_dt_s": 0.0,
        "bound_clip_count": 0.0,
        "bvp_nodes": float(solution.x.size),
        "bvp_max_rms_residual": float(np.max(solution.rms_residuals)),
        "upper_peak_droplet_charge_e": top_peak,
        "lower_peak_droplet_charge_e": bottom_peak,
        "upper_integrated_space_charge_c_m2": top_surface_charge,
        "lower_integrated_space_charge_c_m2": bottom_surface_charge,
        "upper_peak_space_charge_c_m3": top_rho_peak,
        "lower_peak_space_charge_c_m3": bottom_rho_peak,
        "upper_positive_droplet_fraction": upper_positive_fraction,
        "lower_negative_droplet_fraction": lower_negative_fraction,
        "outside_electric_field_v_m": outside_field,
        "cloud_core_electric_field_v_m": inside_field,
        "opposite_droplet_sign_test": sign_pass,
        "opposite_space_charge_sign_test": space_charge_sign_pass,
        "conductivity_field_step_test": conductivity_step_pass,
        "beard2004_cloud_top_reference_e": observed_top_e,
        "beard2004_cloud_base_reference_e": observed_bottom_e,
        "cloud_top_model_minus_observation_e": top_peak - observed_top_e,
        "cloud_base_model_minus_observation_e": bottom_peak - observed_bottom_e,
        "clear_air_n_positive_m3": n_pos_clear,
        "clear_air_n_negative_m3": n_neg_clear,
        "clear_air_neutral_aerosol_m3": neutral_clear,
        "clear_air_s_positive_m3": s_pos_clear,
        "clear_air_s_negative_m3": s_neg_clear,
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
        "droplet_peak_number_m3": model.droplet_peak_m3,
        "paper_reference": "Zhou and Tinsley (2007), JGR 112, D11203",
        "coordinate_convention": "z and E positive downward; fair-weather Jz positive downward",
        "boundary_conditions": (
            "positive-ion concentration specified at upper inflow boundary and "
            "negative-ion concentration specified at lower inflow boundary"
        ),
        "charging_relation": (
            "p=(4*pi*epsilon0*k*T*A/e^2)*ln(D_plus*n_plus/(D_minus*n_minus))"
        ),
        "observation_reference": (
            "Beard et al. (2004), largest mean charges about +85e at cloud top "
            "and -65e at cloud base, as summarized in Zhou and Tinsley (2007)."
        ),
        "model_notes": (
            "Steady one-dimensional Zhou-Tinsley boundary-value formulation. "
            "Drift charging is omitted, matching the paper conclusion that it is small "
            "for the modeled droplet sizes and fields."
        ),
    }
    if progress_callback:
        status = "通过" if sign_pass and space_charge_sign_pass else "未通过"
        progress_callback(1.0, f"论文基准求解完成：上下边界异号检验{status}")
    return SimulationResult(
        z_m=z.copy(),
        snapshot_times_s=np.asarray(sorted(snapshots.keys()), dtype=float),
        snapshots=snapshots,
        diagnostics=diagnostics,
        metadata=metadata,
    )
