from __future__ import annotations

"""Time-dependent layer-cloud charging model based on Zhou & Tinsley (2012).

This module extends the 2007 steady formulation by explicitly integrating the
positive/negative ion concentrations, singly charged aerosol populations, and
mean droplet charge.  It follows the 2012 experiment sequence:

* t=0: ion-aerosol equilibrium, droplets effectively uncharged and 0.01 um.
* 0--600 s: fixed droplet concentration, radius grows linearly.
* 600--5000 s: radius fixed; the first Jz and q are maintained.
* t=5000 s: Jz, or both Jz and q, change as a step.
* 5000--10000 s: approach to the second equilibrium.

The coordinate z and electric field are positive downward.  A constrained
Poisson/current projection is used at each time step, matching the intent of the
spatial iterations in the original IDL programs while adding convergence and
positivity controls.
"""

from dataclasses import asdict
import math
from typing import TYPE_CHECKING, Optional

import numpy as np
from scipy.optimize import least_squares

from .constants import ELEMENTARY_CHARGE, EPSILON_0
from .solver import (
    N_NEG,
    N_POS,
    P_DROP,
    S_NEG,
    S_POS,
    FieldSolution,
    SimulationCancelled,
    SimulationResult,
)

if TYPE_CHECKING:  # pragma: no cover
    from .solver import CancelCallback, CloudChargeModel, ProgressCallback


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


def _fit_relaxation_tau(times: np.ndarray, values: np.ndarray, start_s: float, end_s: float) -> tuple[float, float, float]:
    """Fit y=y_inf+(y0-y_inf)exp(-(t-start)/tau); return tau, y0, y_inf."""
    mask = (times >= start_s) & (times <= end_s) & np.isfinite(values)
    t = np.asarray(times[mask], dtype=float)
    y = np.asarray(values[mask], dtype=float)
    if t.size < 8 or float(np.ptp(y)) < 1.0e-6:
        return math.nan, float(y[0]) if y.size else math.nan, float(y[-1]) if y.size else math.nan
    t = t - t[0]
    y0 = float(y[0])
    tail = max(3, min(y.size // 5, 30))
    y_inf_guess = float(np.mean(y[-tail:]))
    tau_guess = max(1.0, 0.25 * max(float(t[-1]), 1.0))
    scale = max(float(np.ptp(y)), 1.0)

    def residual(params: np.ndarray) -> np.ndarray:
        y_inf, log_tau = params
        tau = math.exp(float(log_tau))
        pred = y_inf + (y0 - y_inf) * np.exp(-t / tau)
        return (pred - y) / scale

    fit = least_squares(
        residual,
        np.array([y_inf_guess, math.log(tau_guess)]),
        loss="soft_l1",
        xtol=1.0e-11,
        ftol=1.0e-11,
        gtol=1.0e-11,
        max_nfev=3000,
    )
    if not fit.success:
        return math.nan, y0, y_inf_guess
    y_inf = float(fit.x[0])
    tau = float(math.exp(fit.x[1]))
    if tau > 20.0 * max(float(t[-1]), 1.0):
        tau = math.nan
    return tau, y0, y_inf


def _paper_reference_case(altitude_km: float, aerosol_m3: float, radius_um: float, q2: float, j2: float) -> dict[str, float | str]:
    """Nearest published 2012 reference values for diagnostics, not calibration."""
    # Values quoted in the paper text/conclusions. Signs use cloud-top positive,
    # cloud-base negative. They are only displayed as reference comparisons.
    if abs(altitude_km - 4.0) < 0.2 and abs(radius_um - 10.0) < 0.3 and aerosol_m3 < 2.0e9:
        return {"case": "2012 Fig.2-4 baseline", "tau_s": 800.0, "top_e": 107.0, "base_e": -117.0}
    if abs(altitude_km - 2.0) < 0.2 and aerosol_m3 < 5.0e9:
        return {"case": "2012 Fig.5 low-aerosol 2 km", "tau_s": 1500.0, "top_e": 185.0, "base_e": -190.0}
    if abs(altitude_km - 2.0) < 0.2 and aerosol_m3 > 1.0e10:
        return {"case": "2012 Fig.6 polluted 2 km", "tau_s": 4100.0, "top_e": 75.0, "base_e": -65.0}
    if abs(altitude_km - 10.0) < 0.2 and q2 < 2.5e7:
        return {"case": "2012 Fig.7 low-latitude 10 km", "tau_s": 450.0, "top_e": 57.0, "base_e": -67.0}
    if abs(altitude_km - 10.0) < 0.2 and q2 >= 2.5e7:
        return {"case": "2012 Fig.8 high-q 10 km", "tau_s": 100.0, "top_e": 20.0, "base_e": -40.0}
    if abs(altitude_km - 4.0) < 0.2 and abs(radius_um - 5.0) < 0.3:
        return {"case": "2012 Fig.10 5 um", "tau_s": 390.0, "top_e": 40.0, "base_e": -40.0}
    if abs(altitude_km - 4.0) < 0.2 and abs(radius_um - 15.0) < 0.3:
        return {"case": "2012 Fig.10 15 um", "tau_s": 1300.0, "top_e": 210.0, "base_e": -210.0}
    return {"case": "custom", "tau_s": math.nan, "top_e": math.nan, "base_e": math.nan}


def run_paper2012_direct(
    model: "CloudChargeModel",
    progress_callback: Optional["ProgressCallback"] = None,
    cancel_callback: Optional["CancelCallback"] = None,
) -> SimulationResult:
    cfg = model.config
    td = cfg.time_dependent

    # Force the published model choices for this mode. The GUI still reports them.
    cfg.physics.charging_mode = "revised"
    cfg.physics.include_drift_charging = True
    cfg.physics.field_mode = "conductivity_only"
    cfg.droplet.growth_time_s = max(cfg.droplet.growth_time_s, 0.0)
    cfg.numerics.duration_s = td.total_duration_s

    # The model object was already created, but initial_state reads q from config.
    cfg.atmosphere.current_density_a_m2 = td.initial_current_density_a_m2
    cfg.atmosphere.ion_pair_production_m3_s = td.initial_ion_pair_production_m3_s

    duration = float(td.total_duration_s)
    step_time = float(td.step_time_s)
    event_times = sorted(
        set(
            [0.0, duration, step_time, cfg.droplet.growth_time_s]
            + [min(duration, max(0.0, float(t))) for t in cfg.numerics.snapshot_times_s]
        )
    )

    state = model.initial_state()
    field = model.solve_field(state)
    state = field.state
    time_s = 0.0
    snapshots: dict[float, dict[str, np.ndarray | float]] = {0.0: model._snapshot(state, 0.0, field)}
    next_snapshot = 1

    upper, _, lower = _boundary_masks(model)
    top_points = np.where(upper & (model.droplet_number_m3 > 0.01 * model.droplet_peak_m3))[0]
    bottom_points = np.where(lower & (model.droplet_number_m3 > 0.01 * model.droplet_peak_m3))[0]
    if top_points.size == 0 or bottom_points.size == 0:
        raise RuntimeError("Cloud boundary masks contain no cloudy grid points.")

    diag_t: list[float] = []
    top_p: list[float] = []
    bottom_p: list[float] = []
    top_dpdt: list[float] = []
    bottom_dpdt: list[float] = []
    top_spos: list[float] = []
    top_sneg: list[float] = []
    top_npos: list[float] = []
    top_nneg: list[float] = []
    max_field: list[float] = []
    current_residual: list[float] = []
    column_charge: list[float] = []
    radius_history: list[float] = []
    current_history: list[float] = []
    ion_rate_history: list[float] = []
    p_profile_history: list[np.ndarray] = []
    dpdt_profile_history: list[np.ndarray] = []
    spos_profile_history: list[np.ndarray] = []
    sneg_profile_history: list[np.ndarray] = []
    npos_profile_history: list[np.ndarray] = []
    nneg_profile_history: list[np.ndarray] = []

    diagnostic_interval = max(float(cfg.numerics.diagnostics_interval_s), cfg.numerics.max_dt_s)
    next_diag = 0.0
    step_count = 0
    min_dt = cfg.numerics.max_dt_s
    max_dt = 0.0
    model._clip_count = 0

    clear_cache: dict[float, tuple[float, float, float, float]] = {}

    def set_phase(phase2: bool) -> tuple[float, float]:
        if phase2:
            j = td.second_current_density_a_m2
            q = td.second_ion_pair_production_m3_s
        else:
            j = td.initial_current_density_a_m2
            q = td.initial_ion_pair_production_m3_s
        cfg.atmosphere.current_density_a_m2 = float(j)
        cfg.atmosphere.ion_pair_production_m3_s = float(q)
        return float(j), float(q)

    def enforce_inflow(local_state: np.ndarray) -> np.ndarray:
        """Positive ions enter at the upper boundary; negative ions at the lower."""
        q_key = float(cfg.atmosphere.ion_pair_production_m3_s)
        if q_key not in clear_cache:
            clear_cache[q_key] = model._homogeneous_equilibrium_corrected()
        npos_clear, nneg_clear, _spos_clear, _sneg_clear = clear_cache[q_key]
        local_state = np.asarray(local_state, dtype=float).copy()
        local_state[N_POS, 0] = npos_clear
        local_state[N_NEG, -1] = nneg_clear
        return model._enforce_bounds(local_state)

    def constrained_field(local_state: np.ndarray) -> FieldSolution:
        """Enforce Ohm and Gauss constraints by an ion-difference projection.

        E is first obtained from Jz=e(mu+ n+ + mu- n-)E.  Its spatial
        gradient defines the charge required by Gauss's law.  The positive and
        negative ion concentrations are then corrected in opposite directions
        so that their conductivity is unchanged but their charge difference
        supplies the missing space charge.  This represents the rapid spatial
        electrode-effect iteration in the 2012 IDL model.
        """
        local_state = enforce_inflow(local_state)
        mu_pos = model.atmosphere.positive_mobility_m2_v_s
        mu_neg = model.atmosphere.negative_mobility_m2_v_s
        j_target = float(cfg.atmosphere.current_density_a_m2)
        residual = math.inf
        used = 0
        for iteration in range(max(1, cfg.numerics.field_iterations)):
            sigma_number = np.maximum(
                mu_pos * local_state[N_POS] + mu_neg * local_state[N_NEG], 1.0e-30
            )
            electric_field = j_target / (ELEMENTARY_CHARGE * sigma_number)
            dfield_dz = np.gradient(electric_field, model.z, edge_order=2)
            required_charge_number = EPSILON_0 * dfield_dz / ELEMENTARY_CHARGE
            other_charge_number = (
                local_state[S_POS] - local_state[S_NEG]
                + local_state[P_DROP] * model.droplet_number_m3
            )
            ion_difference = local_state[N_POS] - local_state[N_NEG]
            mismatch = required_charge_number - (ion_difference + other_charge_number)
            scale = max(float(np.max(np.abs(required_charge_number))), 1.0)
            residual = float(np.max(np.abs(mismatch)) / scale)
            used = iteration + 1
            if residual <= cfg.numerics.field_tolerance:
                break
            # Keep sigma_number fixed: mu+*dn+ + mu-*dn- = 0,
            # while dn+ - dn- = mismatch.
            dn_pos = mu_neg * mismatch / (mu_pos + mu_neg)
            dn_neg = -mu_pos * mismatch / (mu_pos + mu_neg)
            relax = cfg.numerics.field_relaxation
            # Do not change prescribed inflow components at their boundaries.
            dn_pos[0] = 0.0
            dn_neg[-1] = 0.0
            # Positivity limiter before applying the correction.
            dn_pos = np.maximum(dn_pos, -0.90 * local_state[N_POS])
            dn_neg = np.maximum(dn_neg, -0.90 * local_state[N_NEG])
            local_state[N_POS] += relax * dn_pos
            local_state[N_NEG] += relax * dn_neg
            local_state = enforce_inflow(local_state)

        sigma_number = np.maximum(mu_pos * local_state[N_POS] + mu_neg * local_state[N_NEG], 1.0e-30)
        electric_field = j_target / (ELEMENTARY_CHARGE * sigma_number)
        dfield_dz = np.gradient(electric_field, model.z, edge_order=2)
        charge_density = ELEMENTARY_CHARGE * (
            local_state[N_POS] - local_state[N_NEG]
            + local_state[S_POS] - local_state[S_NEG]
            + local_state[P_DROP] * model.droplet_number_m3
        )
        current = ELEMENTARY_CHARGE * sigma_number * electric_field
        current_residual = float(np.max(np.abs(current - j_target)) / max(abs(j_target), 1.0e-30))
        return FieldSolution(
            state=local_state,
            electric_field_v_m=electric_field,
            dfield_dz_v_m2=dfield_dz,
            charge_density_c_m3=charge_density,
            current_density_a_m2=current,
            projection_iterations=used,
            relative_current_residual=current_residual,
        )

    def record() -> None:
        nonlocal field
        _, terms = model.rhs(state, time_s, field, return_terms=True)
        charge_rate = np.asarray(terms["droplet_charge_rate_e_s"], dtype=float) if terms else np.zeros_like(model.z)
        ip = top_points[np.argmax(state[P_DROP, top_points])]
        im = bottom_points[np.argmin(state[P_DROP, bottom_points])]
        diag_t.append(float(time_s))
        top_p.append(float(state[P_DROP, ip]))
        bottom_p.append(float(state[P_DROP, im]))
        top_dpdt.append(float(charge_rate[ip]))
        bottom_dpdt.append(float(charge_rate[im]))
        top_spos.append(float(state[S_POS, ip]))
        top_sneg.append(float(state[S_NEG, ip]))
        top_npos.append(float(state[N_POS, ip]))
        top_nneg.append(float(state[N_NEG, ip]))
        max_field.append(float(np.max(np.abs(field.electric_field_v_m))))
        current_residual.append(float(field.relative_current_residual))
        column_charge.append(float(np.trapezoid(field.charge_density_c_m3, model.z)))
        radius_history.append(float(model._snapshot(state, time_s, field)["droplet_radius_m"]))
        current_history.append(float(cfg.atmosphere.current_density_a_m2))
        ion_rate_history.append(float(cfg.atmosphere.ion_pair_production_m3_s))
        p_profile_history.append(state[P_DROP].copy())
        dpdt_profile_history.append(charge_rate.copy())
        spos_profile_history.append(state[S_POS].copy())
        sneg_profile_history.append(state[S_NEG].copy())
        npos_profile_history.append(state[N_POS].copy())
        nneg_profile_history.append(state[N_NEG].copy())

    set_phase(False)
    field = constrained_field(state)
    state = field.state
    record()
    if progress_callback:
        progress_callback(0.0, "2012时间依赖模型初始化：离子-气溶胶平衡，云滴开始生长")

    while time_s < duration - 1.0e-12:
        if cancel_callback and cancel_callback():
            raise SimulationCancelled("Simulation cancelled by user.")

        phase2 = time_s >= step_time - 1.0e-10
        j_now, q_now = set_phase(phase2)
        field = constrained_field(state)
        state = field.state
        rhs1, _ = model.rhs(state, time_s, field)

        proposed = min(cfg.numerics.max_dt_s, duration - time_s)
        # Do not cross the Jz/q discontinuity or a requested snapshot.
        future_events = [t for t in event_times if t > time_s + 1.0e-10]
        if future_events:
            proposed = min(proposed, future_events[0] - time_s)
        # The published IDL integration used a fixed one-second increment.
        # Reaction stiffness is controlled by the predictor-corrector and the
        # non-negativity projection rather than allowing the step to collapse.
        dt = proposed
        if dt <= 0:
            raise RuntimeError("Non-positive time step in 2012 solver.")

        # Keep the phase constant over a step that terminates exactly at the discontinuity.
        cfg.atmosphere.current_density_a_m2 = j_now
        cfg.atmosphere.ion_pair_production_m3_s = q_now
        stage1 = enforce_inflow(model._enforce_bounds(state + dt * rhs1))
        stage1_field = constrained_field(stage1)
        stage1 = stage1_field.state
        rhs2, _ = model.rhs(stage1, time_s + dt, stage1_field)
        state = enforce_inflow(model._enforce_bounds(0.5 * state + 0.5 * (stage1 + dt * rhs2)))
        state = enforce_inflow(model._apply_conservative_smoothing(state))
        field = constrained_field(state)
        state = field.state

        time_s += dt
        step_count += 1
        min_dt = min(min_dt, dt)
        max_dt = max(max_dt, dt)

        while next_snapshot < len(event_times) and time_s >= event_times[next_snapshot] - 1.0e-8:
            snap_t = float(event_times[next_snapshot])
            snapshots[snap_t] = model._snapshot(state, snap_t, field)
            next_snapshot += 1

        if time_s >= next_diag - 1.0e-9:
            # At exactly 5000 s the stored state is the end of the first phase.
            set_phase(time_s > step_time + 1.0e-9)
            field = constrained_field(state)
            state = field.state
            record()
            next_diag += diagnostic_interval

        if progress_callback and (step_count == 1 or step_count % 100 == 0):
            phase_name = "第二阶段" if phase2 else ("云滴增长" if time_s <= cfg.droplet.growth_time_s else "启动趋稳")
            progress_callback(
                min(1.0, time_s / duration),
                f"{phase_name}: t={time_s:.0f}s, Jz={j_now/1e-12:.2f} pA m⁻², max|p|={np.max(np.abs(state[P_DROP])):.1f}e",
            )

    set_phase(True)
    final_field = constrained_field(state)
    state = final_field.state
    snapshots[duration] = model._snapshot(state, duration, final_field)
    field = final_field
    time_s = duration
    if not diag_t or abs(diag_t[-1] - duration) > 1.0e-6:
        record()

    times = np.asarray(diag_t)
    p_history = np.stack(p_profile_history, axis=0)
    rate_history = np.stack(dpdt_profile_history, axis=0)
    spos_history = np.stack(spos_profile_history, axis=0)
    sneg_history = np.stack(sneg_profile_history, axis=0)
    npos_history = np.stack(npos_profile_history, axis=0)
    nneg_history = np.stack(nneg_profile_history, axis=0)
    final_profile_for_index = p_history[-1]
    fixed_top_index = int(top_points[np.argmax(final_profile_for_index[top_points])])
    fixed_bottom_index = int(bottom_points[np.argmin(final_profile_for_index[bottom_points])])
    top_series = p_history[:, fixed_top_index]
    bottom_series = p_history[:, fixed_bottom_index]
    top_rate_series = rate_history[:, fixed_top_index]
    bottom_rate_series = rate_history[:, fixed_bottom_index]
    top_spos_series = spos_history[:, fixed_top_index]
    top_sneg_series = sneg_history[:, fixed_top_index]
    top_npos_series = npos_history[:, fixed_top_index]
    top_nneg_series = nneg_history[:, fixed_top_index]
    startup_start = max(float(cfg.droplet.growth_time_s), 0.0)
    tau_start_top, _, _ = _fit_relaxation_tau(times, top_series, startup_start, min(step_time, duration))
    tau_start_bottom, _, _ = _fit_relaxation_tau(times, bottom_series, startup_start, min(step_time, duration))
    tau_step_top, _, top_inf = _fit_relaxation_tau(times, top_series, step_time, duration)
    tau_step_bottom, _, bottom_inf = _fit_relaxation_tau(times, bottom_series, step_time, duration)

    final = snapshots[duration]
    final_p = np.asarray(final["droplet_charge_e"])
    final_rho = np.asarray(final["charge_density_c_m3"])
    final_top_peak = float(np.max(final_p[top_points]))
    final_bottom_peak = float(np.min(final_p[bottom_points]))
    upper_charge = float(np.trapezoid(final_rho[upper], model.z[upper]))
    lower_charge = float(np.trapezoid(final_rho[lower], model.z[lower]))
    numeric_dfield = np.gradient(np.asarray(final["electric_field_v_m"]), model.z, edge_order=2)
    poisson_residual = numeric_dfield - final_rho / EPSILON_0

    reference = _paper_reference_case(
        cfg.atmosphere.altitude_km,
        cfg.aerosol.total_number_m3,
        cfg.droplet.target_radius_m / 1.0e-6,
        td.second_ion_pair_production_m3_s,
        td.second_current_density_a_m2,
    )

    diagnostics: dict[str, np.ndarray | float | str | bool] = {
        "solver_mode": "paper2012_direct",
        "time_s": times,
        "top_droplet_charge_e": top_series,
        "bottom_droplet_charge_e": bottom_series,
        "top_droplet_charging_rate_e_s": top_rate_series,
        "bottom_droplet_charging_rate_e_s": bottom_rate_series,
        "top_positive_aerosol_m3": top_spos_series,
        "top_negative_aerosol_m3": top_sneg_series,
        "top_positive_ion_m3": top_npos_series,
        "top_negative_ion_m3": top_nneg_series,
        "fixed_top_index": float(fixed_top_index),
        "fixed_bottom_index": float(fixed_bottom_index),
        "fixed_top_depth_m": float(model.z[fixed_top_index]),
        "fixed_bottom_depth_m": float(model.z[fixed_bottom_index]),
        "max_abs_electric_field_v_m": np.asarray(max_field),
        "relative_current_residual": np.asarray(current_residual),
        "column_net_charge_c_m2": np.asarray(column_charge),
        "droplet_radius_m": np.asarray(radius_history),
        "scheduled_current_density_a_m2": np.asarray(current_history),
        "scheduled_ion_pair_production_m3_s": np.asarray(ion_rate_history),
        "startup_tau_top_s": tau_start_top,
        "startup_tau_bottom_s": tau_start_bottom,
        "post_step_tau_top_s": tau_step_top,
        "post_step_tau_bottom_s": tau_step_bottom,
        "post_step_fitted_top_equilibrium_e": top_inf,
        "post_step_fitted_bottom_equilibrium_e": bottom_inf,
        "upper_peak_droplet_charge_e": final_top_peak,
        "lower_peak_droplet_charge_e": final_bottom_peak,
        "upper_integrated_space_charge_c_m2": upper_charge,
        "lower_integrated_space_charge_c_m2": lower_charge,
        "opposite_droplet_sign_test": bool(final_top_peak > 0 and final_bottom_peak < 0),
        "opposite_space_charge_sign_test": bool(upper_charge > 0 and lower_charge < 0),
        "final_poisson_residual_v_m2": poisson_residual,
        "final_max_relative_current_residual": float(final_field.relative_current_residual),
        "time_steps": float(step_count),
        "min_dt_s": float(min_dt),
        "max_dt_s": float(max_dt),
        "bound_clip_count": float(model._clip_count),
        "paper_reference_case": str(reference["case"]),
        "paper_reference_tau_s": float(reference["tau_s"]),
        "paper_reference_top_e": float(reference["top_e"]),
        "paper_reference_base_e": float(reference["base_e"]),
    }
    if math.isfinite(float(reference["tau_s"])):
        diagnostics["tau_top_minus_paper_s"] = tau_step_top - float(reference["tau_s"])
        diagnostics["top_charge_minus_paper_e"] = final_top_peak - float(reference["top_e"])
        diagnostics["base_charge_minus_paper_e"] = final_bottom_peak - float(reference["base_e"])

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
        "paper_reference": "Zhou and Tinsley (2012), Advances in Space Research 50, 828-842",
        "coordinate_convention": "z and E positive downward; fair-weather Jz positive downward",
        "time_sequence": {
            "growth_end_s": cfg.droplet.growth_time_s,
            "step_time_s": step_time,
            "duration_s": duration,
            "initial_Jz_a_m2": td.initial_current_density_a_m2,
            "second_Jz_a_m2": td.second_current_density_a_m2,
            "initial_q_m3_s": td.initial_ion_pair_production_m3_s,
            "second_q_m3_s": td.second_ion_pair_production_m3_s,
        },
        "model_notes": (
            "Time-dependent integration of n+, n-, S+, S-, and mean droplet charge p. "
            "Diffusion and first-order drift charging are coupled. At every time step a "
            "Poisson/current projection iterates the vertical field and ion conductivity."
        ),
        "comparison_note": (
            "Published reference values are diagnostics only and are not used to tune or force the solution."
        ),
    }
    if progress_callback:
        progress_callback(
            1.0,
            f"2012模拟完成：云顶{final_top_peak:+.1f}e，云底{final_bottom_peak:+.1f}e，响应τ≈{tau_step_top:.0f}s",
        )
    return SimulationResult(
        z_m=model.z.copy(),
        snapshot_times_s=np.asarray(sorted(snapshots.keys()), dtype=float),
        snapshots=snapshots,
        diagnostics=diagnostics,
        metadata=metadata,
    )
