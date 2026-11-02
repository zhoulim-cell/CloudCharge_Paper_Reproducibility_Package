from __future__ import annotations

from dataclasses import asdict
import math
from typing import Callable, Optional

import numpy as np
from scipy.linalg import solve_banded

from .constants import ELEMENTARY_CHARGE, EPSILON_0, PI
from .frontier_attachment import MeasuredAttachmentCorrection
from .frontier_config import FrontierConfig
from .frontier_numerics import (
    build_radius_grid,
    charge_moments,
    charging_rates,
    gaussian_charge_weights,
    implicit_charge_step,
    lognormal_bin_weights,
    radius_advection_step,
    terminal_velocity_stokes,
    transport_dt_limit,
    transport_step,
)
from .frontier_scavenging import ScavengingDiagnostics, apply_electroscavenging
from .solver import CloudChargeModel, SimulationCancelled, SimulationResult


ProgressCallback = Callable[[float, str], None]
CancelCallback = Callable[[], bool]


def _forcing_value(mode: str, t: float, initial: float, final: float, cfg) -> float:
    if mode == "constant":
        return float(initial)
    if mode == "step":
        return float(initial if t < cfg.step_time_s else final)
    if mode == "sinusoid":
        return float(initial * (1.0 + cfg.sinusoid_fractional_amplitude * math.sin(2.0 * PI * t / cfg.sinusoid_period_s)))
    if mode == "forbush":
        if t < cfg.step_time_s:
            return float(initial)
        elapsed = t - cfg.step_time_s
        return float(initial * (1.0 - cfg.forbush_drop_fraction * math.exp(-elapsed / cfg.forbush_recovery_time_s)))
    raise ValueError(f"Unknown forcing mode {mode}")


def _local_ion_equilibrium(
    q_m3_s: float,
    alpha_m3_s: float,
    d_pos: float,
    d_neg: float,
    droplet_surface_radius_m_m3: np.ndarray,
    aerosol_surface_radius_m_m3: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Bipolar equilibrium including size-resolved particle attachment sinks."""
    lpos = 4.0 * PI * d_pos * (droplet_surface_radius_m_m3 + aerosol_surface_radius_m_m3)
    lneg = 4.0 * PI * d_neg * (droplet_surface_radius_m_m3 + aerosol_surface_radius_m_m3)
    lpos = np.maximum(lpos, 1.0e-12)
    lneg = np.maximum(lneg, 1.0e-12)
    ratio = lneg / lpos  # n+ / n- from equal positive/negative particle loss fluxes
    disc = lneg**2 + 4.0 * alpha_m3_s * ratio * q_m3_s
    nneg = (-lneg + np.sqrt(np.maximum(disc, 0.0))) / (2.0 * alpha_m3_s * ratio)
    npos = ratio * nneg
    return np.maximum(npos, 1.0), np.maximum(nneg, 1.0)


def _cumulative_trapezoid(values: np.ndarray, z: np.ndarray) -> np.ndarray:
    out = np.zeros_like(values)
    out[1:] = np.cumsum(0.5 * (values[1:] + values[:-1]) * np.diff(z))
    return out


def _field_from_charge(rho: np.ndarray, z: np.ndarray, reference: np.ndarray) -> np.ndarray:
    integral = _cumulative_trapezoid(rho / EPSILON_0, z)
    offset = float(np.mean(reference - integral))
    return integral + offset


def _non_displacement_current(
    npos: np.ndarray,
    nneg: np.ndarray,
    electric: np.ndarray,
    z: np.ndarray,
    mu_pos: float,
    mu_neg: float,
    d_pos: float,
    d_neg: float,
    air_velocity_down_m_s: float,
    ion_eddy_diffusivity_m2_s: float,
    particle_eddy_diffusivity_m2_s: float,
    aerosol_moments: dict[str, np.ndarray],
    droplet: np.ndarray,
    droplet_radii_m: np.ndarray,
    droplet_charges_e: np.ndarray,
    include_droplet_sedimentation: bool,
    include_diffusion: bool,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """All charge-transport currents except displacement current.

    Including turbulent/advective particle-charge transport is essential once
    cloud-edge mixing is enabled; otherwise charge continuity and Gauss law
    cannot remain compatible with the imposed total current.
    """
    sigma_current = ELEMENTARY_CHARGE * (mu_pos * npos + mu_neg * nneg)
    ion_drift = sigma_current * electric
    ion_other = ELEMENTARY_CHARGE * air_velocity_down_m_s * (npos - nneg)
    if include_diffusion:
        ion_other += ELEMENTARY_CHARGE * (
            -(d_pos + ion_eddy_diffusivity_m2_s) * np.gradient(npos, z, edge_order=2)
            + (d_neg + ion_eddy_diffusivity_m2_s) * np.gradient(nneg, z, edge_order=2)
        )
    qa = aerosol_moments["charge_number_density_e_m3"]
    aerosol_current = ELEMENTARY_CHARGE * (
        air_velocity_down_m_s * qa
        - particle_eddy_diffusivity_m2_s * np.gradient(qa, z, edge_order=2)
    )
    qdr = np.sum(droplet * droplet_charges_e[None, None, :], axis=2)
    velocities = np.full(droplet_radii_m.size, air_velocity_down_m_s, dtype=float)
    if include_droplet_sedimentation:
        velocities += terminal_velocity_stokes(droplet_radii_m)
    qd = np.sum(qdr, axis=1)
    droplet_current = ELEMENTARY_CHARGE * (
        np.sum(qdr * velocities[None, :], axis=1)
        - particle_eddy_diffusivity_m2_s * np.gradient(qd, z, edge_order=2)
    )
    total = ion_drift + ion_other + aerosol_current + droplet_current
    return total, {
        "sigma_current_s_m": sigma_current,
        "ion_drift_current_a_m2": ion_drift,
        "ion_other_current_a_m2": ion_other,
        "aerosol_charge_current_a_m2": aerosol_current,
        "droplet_charge_current_a_m2": droplet_current,
        "other_current_a_m2": ion_other + aerosol_current + droplet_current,
    }


def _charge_density(
    npos: np.ndarray,
    nneg: np.ndarray,
    aerosol_moments: dict[str, np.ndarray],
    droplet_moments: dict[str, np.ndarray],
) -> np.ndarray:
    return ELEMENTARY_CHARGE * (
        npos
        - nneg
        + aerosol_moments["charge_number_density_e_m3"]
        + droplet_moments["charge_number_density_e_m3"]
    )




def _solve_field_variational(
    electric_old: np.ndarray,
    rho: np.ndarray,
    sigma_current: np.ndarray,
    diffusion_current: np.ndarray,
    jtarget: float,
    dt: float,
    dz: float,
    gauss_weight_fraction: float,
    boundary_field_v_m: float | None = None,
    boundary_anchor_weight: float = 0.0,
) -> np.ndarray:
    """Joint Maxwell-current/Gauss least-squares field update.

    The current equation is backward Euler,
      (eps0/dt + sigma) E_new = Jz - Jdiff + eps0 E_old/dt,
    while Gauss law is imposed on cell faces.  A dimensionless weighted
    least-squares solve avoids the mutually inconsistent sequential
    current update and Poisson projection used in many legacy models.
    """
    n = electric_old.size
    lam = float(np.clip(gauss_weight_fraction, 0.0, 1.0))
    acoef = EPSILON_0 / dt + sigma_current
    bcoef = jtarget - diffusion_current + EPSILON_0 * electric_old / dt
    jscale = max(abs(jtarget), 1.0e-13)
    gface = 0.5 * (rho[:-1] + rho[1:]) / EPSILON_0
    gscale = max(float(np.sqrt(np.mean(gface**2))) if gface.size else 0.0, 0.1)
    wj = max(1.0 - lam, 1.0e-6) / (jscale * jscale)
    wg = max(lam, 1.0e-6) / (gscale * gscale)
    diag = wj * acoef * acoef
    rhs = wj * acoef * bcoef
    off = np.zeros(n - 1, dtype=float)
    if n > 1:
        c = wg / (dz * dz)
        diag[:-1] += c
        diag[1:] += c
        off[:] = -c
        rhs[:-1] += -wg * gface / dz
        rhs[1:] += wg * gface / dz
    if boundary_field_v_m is not None and boundary_anchor_weight > 0.0:
        escale = max(abs(float(boundary_field_v_m)), 10.0)
        wb = float(boundary_anchor_weight) / (escale * escale)
        diag[0] += wb
        diag[-1] += wb
        rhs[0] += wb * float(boundary_field_v_m)
        rhs[-1] += wb * float(boundary_field_v_m)
    ab = np.zeros((3, n), dtype=float)
    ab[1] = diag
    if n > 1:
        ab[0, 1:] = off
        ab[2, :-1] = off
    return solve_banded((1, 1), ab, rhs, check_finite=False)

def _reaction_dt_limit(
    npos: np.ndarray,
    nneg: np.ndarray,
    loss_pos: np.ndarray,
    loss_neg: np.ndarray,
    recombination: np.ndarray,
    proposed: float,
    safety: float,
) -> float:
    sink_pos = recombination + loss_pos
    sink_neg = recombination + loss_neg
    dt = proposed
    mask = sink_pos > 0
    if np.any(mask):
        dt = min(dt, float(np.min(safety * npos[mask] / sink_pos[mask])))
    mask = sink_neg > 0
    if np.any(mask):
        dt = min(dt, float(np.min(safety * nneg[mask] / sink_neg[mask])))
    return max(dt, 1.0e-8)


def _initialise(cfg: FrontierConfig, model: CloudChargeModel):
    # Droplet spectrum.
    dr, dre = build_radius_grid(
        cfg.base.spectrum.radius_min_m,
        cfg.base.spectrum.radius_max_m,
        cfg.base.spectrum.radius_bins,
        cfg.base.spectrum.radius_spacing,
    )
    dq = np.arange(cfg.base.spectrum.charge_min_e, cfg.base.spectrum.charge_max_e + 1, dtype=float)
    dw_r = lognormal_bin_weights(dre, cfg.base.droplet.initial_radius_m, cfg.base.spectrum.geometric_std)
    dw_q = gaussian_charge_weights(dq, cfg.base.spectrum.initial_charge_std_e, neutral=True)
    droplet = model.droplet_number_m3[:, None, None] * dw_r[None, :, None] * dw_q[None, None, :]

    # Aerosol spectrum.
    ac = cfg.aerosol_spectrum
    ar, are = build_radius_grid(ac.radius_min_m, ac.radius_max_m, ac.radius_bins, "log")
    aq = np.arange(ac.charge_min_e, ac.charge_max_e + 1, dtype=float)
    aw_r = lognormal_bin_weights(are, ac.geometric_mean_radius_m, ac.geometric_std)
    aw_q = gaussian_charge_weights(aq, ac.initial_charge_std_e, neutral=False)
    aerosol = cfg.base.aerosol.total_number_m3 * aw_r[None, :, None] * aw_q[None, None, :]
    aerosol = np.broadcast_to(aerosol, (model.z.size, ar.size, aq.size)).copy()

    dm = charge_moments(droplet, dr, dq)
    am = charge_moments(aerosol, ar, aq)
    drop_surface = np.sum(dm["size_number_m3"] * dr[None, :], axis=1)
    aero_surface = np.sum(am["size_number_m3"] * ar[None, :], axis=1)
    npos, nneg = _local_ion_equilibrium(
        cfg.forcing.ionization_initial_m3_s,
        cfg.base.atmosphere.recombination_m3_s,
        model.atmosphere.positive_diffusivity_m2_s,
        model.atmosphere.negative_diffusivity_m2_s,
        drop_surface,
        aero_surface,
    )
    # Bipolar aerosol equilibrium carries the compensating charge required
    # for a neutral homogeneous ion-aerosol mixture.  This removes the
    # artificial domain-wide positive space charge that would result from
    # starting the aerosol spectrum symmetric while mu-/D- > mu+/D+.
    qbar_a = -(npos - nneg) / max(cfg.base.aerosol.total_number_m3, 1.0)
    aerosol = np.empty((model.z.size, ar.size, aq.size), dtype=float)
    for iz, qmean in enumerate(qbar_a):
        wq = np.exp(-0.5 * ((aq - qmean) / ac.initial_charge_std_e) ** 2)
        wq /= np.sum(wq)
        aerosol[iz] = cfg.base.aerosol.total_number_m3 * aw_r[:, None] * wq[None, :]
    am = charge_moments(aerosol, ar, aq)
    # Newly formed cloud: start with the clear-air field and let displacement
    # current build the cloud-interior field on the dielectric relaxation time.
    clear_npos, clear_nneg = _local_ion_equilibrium(
        cfg.forcing.ionization_initial_m3_s,
        cfg.base.atmosphere.recombination_m3_s,
        model.atmosphere.positive_diffusivity_m2_s,
        model.atmosphere.negative_diffusivity_m2_s,
        np.zeros_like(model.z),
        np.full_like(model.z, aero_surface[0]),
    )
    clear_sigma_number = (
        model.atmosphere.positive_mobility_m2_v_s * clear_npos[0]
        + model.atmosphere.negative_mobility_m2_v_s * clear_nneg[0]
    )
    e0 = cfg.forcing.current_density_initial_a_m2 / (ELEMENTARY_CHARGE * clear_sigma_number)
    electric = np.full_like(model.z, e0)
    return droplet, aerosol, dr, dre, dq, ar, are, aq, npos, nneg, clear_npos, clear_nneg, electric


def _snapshot(
    model: CloudChargeModel,
    time_s: float,
    droplet: np.ndarray,
    aerosol: np.ndarray,
    dr: np.ndarray,
    dre: np.ndarray,
    dq: np.ndarray,
    ar: np.ndarray,
    are: np.ndarray,
    aq: np.ndarray,
    npos: np.ndarray,
    nneg: np.ndarray,
    electric: np.ndarray,
    rho: np.ndarray,
    jcond: np.ndarray,
    jdisp: np.ndarray,
    jtarget: float,
) -> dict[str, np.ndarray | float]:
    dm = charge_moments(droplet, dr, dq)
    am = charge_moments(aerosol, ar, aq)
    return {
        "time_s": float(time_s),
        "n_positive_m3": npos.copy(),
        "n_negative_m3": nneg.copy(),
        "electric_field_v_m": electric.copy(),
        "charge_density_c_m3": rho.copy(),
        "conductive_current_a_m2": jcond.copy(),
        "non_displacement_current_a_m2": jcond.copy(),
        "displacement_current_a_m2": jdisp.copy(),
        "total_current_a_m2": (jcond + jdisp).copy(),
        "target_current_a_m2": float(jtarget),
        "droplet_number_m3": dm["number_m3"].copy(),
        "droplet_mean_radius_m": dm["mean_radius_m"].copy(),
        "droplet_charge_e": dm["mean_charge_e"].copy(),
        "droplet_charge_std_e": dm["charge_std_e"].copy(),
        "droplet_positive_fraction": dm["positive_fraction"].copy(),
        "droplet_negative_fraction": dm["negative_fraction"].copy(),
        "aerosol_number_m3": am["number_m3"].copy(),
        "aerosol_mean_radius_m": am["mean_radius_m"].copy(),
        "aerosol_charge_e": am["mean_charge_e"].copy(),
        "aerosol_charge_std_e": am["charge_std_e"].copy(),
        "droplet_radius_centers_m": dr.copy(),
        "droplet_radius_edges_m": dre.copy(),
        "droplet_charge_states_e": dq.copy(),
        "aerosol_radius_centers_m": ar.copy(),
        "aerosol_radius_edges_m": are.copy(),
        "aerosol_charge_states_e": aq.copy(),
        "droplet_spectrum_m3": droplet.astype(np.float32, copy=True),
        "aerosol_spectrum_m3": aerosol.astype(np.float32, copy=True),
    }


def run_frontier_model(
    cfg: FrontierConfig,
    progress_callback: Optional[ProgressCallback] = None,
    cancel_callback: Optional[CancelCallback] = None,
) -> SimulationResult:
    """Charge-resolved aerosol–ion–droplet model with dynamic electric field.

    Innovations beyond Zhou & Tinsley (2007, 2012): two particle size-charge
    spectra, finite displacement current, vertical turbulent transport,
    optional condensation and charge-resolved electroscavenging, and direct
    observation/inversion-ready diagnostics.
    """
    cfg.validate()
    # Keep the base model on the requested frontier duration and resolution.
    cfg.base.numerics.duration_s = cfg.numerics.duration_s
    model = CloudChargeModel(cfg.base)
    (
        droplet,
        aerosol,
        dr,
        dre,
        dq,
        ar,
        are,
        aq,
        npos,
        nneg,
        clear_npos,
        clear_nneg,
        electric,
    ) = _initialise(cfg, model)
    correction = MeasuredAttachmentCorrection(cfg.aerosol_spectrum.measured_attachment_table)
    aerosol_reservoir_top = aerosol[0].copy()
    aerosol_reservoir_bottom = aerosol[-1].copy()

    z = model.z
    dz = model.dz
    mu_pos = model.atmosphere.positive_mobility_m2_v_s
    mu_neg = model.atmosphere.negative_mobility_m2_v_s
    d_pos = model.atmosphere.positive_diffusivity_m2_s
    d_neg = model.atmosphere.negative_diffusivity_m2_s
    temperature = model.atmosphere.temperature_k
    alpha = cfg.base.atmosphere.recombination_m3_s
    duration = cfg.numerics.duration_s
    requested = sorted(set(float(x) for x in cfg.numerics.snapshot_times_s if 0 <= x <= duration) | {0.0, duration})
    snapshots: dict[float, dict[str, np.ndarray | float]] = {}

    t = 0.0
    step = 0
    next_snapshot = 0
    next_diag = 0.0
    previous_electric = electric.copy()
    zero = np.zeros_like(z)
    last_jdisp = zero.copy()
    last_scav = ScavengingDiagnostics(zero.copy(), zero.copy(), np.ones_like(z))
    scavenging_elapsed_s = 0.0

    diag_time: list[float] = []
    diag_top_q: list[float] = []
    diag_base_q: list[float] = []
    diag_top_aq: list[float] = []
    diag_base_aq: list[float] = []
    diag_max_e: list[float] = []
    diag_gauss: list[float] = []
    diag_gauss_relative: list[float] = []
    diag_gauss_boundary: list[float] = []
    diag_current: list[float] = []
    diag_total_charge: list[float] = []
    diag_displacement_fraction: list[float] = []
    diag_scavenged: list[float] = []
    diag_mult: list[float] = []
    forcing_j_hist: list[float] = []
    forcing_q_hist: list[float] = []

    cloud_start = cfg.base.domain.cloud_start_m
    top_mask = (z >= cloud_start) & (z <= cloud_start + cfg.base.domain.edge_ramp_m)
    bottom_start = cloud_start + cfg.base.domain.edge_ramp_m + cfg.base.domain.core_width_m
    base_mask = (z >= bottom_start) & (z <= bottom_start + cfg.base.domain.edge_ramp_m)

    def diagnostics(jtarget: float, qtarget: float, jcond: np.ndarray, jdisp: np.ndarray, rho: np.ndarray) -> None:
        dm = charge_moments(droplet, dr, dq)
        am = charge_moments(aerosol, ar, aq)
        gauss = np.gradient(electric, z, edge_order=2) - rho / EPSILON_0
        total_current = jcond + jdisp
        diag_time.append(float(t))
        diag_top_q.append(float(np.max(dm["mean_charge_e"][top_mask])))
        diag_base_q.append(float(np.min(dm["mean_charge_e"][base_mask])))
        diag_top_aq.append(float(np.max(am["mean_charge_e"][top_mask])))
        diag_base_aq.append(float(np.min(am["mean_charge_e"][base_mask])))
        diag_max_e.append(float(np.max(np.abs(electric))))
        interior = slice(2, -2) if z.size > 5 else slice(None)
        gauss_scale = max(float(np.max(np.abs((rho / EPSILON_0)[interior]))), 1.0e-3)
        diag_gauss.append(float(np.max(np.abs(gauss))))
        diag_gauss_relative.append(float(np.max(np.abs(gauss[interior])) / gauss_scale))
        if z.size > 5:
            boundary_values = np.concatenate((gauss[:2], gauss[-2:]))
            diag_gauss_boundary.append(float(np.max(np.abs(boundary_values))))
        else:
            diag_gauss_boundary.append(float(np.max(np.abs(gauss))))
        diag_current.append(float(np.max(np.abs(total_current - jtarget)) / max(abs(jtarget), 1.0e-30)))
        diag_total_charge.append(float(np.trapezoid(rho, z)))
        diag_displacement_fraction.append(float(np.mean(np.abs(jdisp)) / max(abs(jtarget), 1.0e-30)))
        diag_scavenged.append(float(np.trapezoid(last_scav.aerosol_removed_m3, z)))
        diag_mult.append(float(np.mean(last_scav.mean_electrostatic_multiplier)))
        forcing_j_hist.append(float(jtarget))
        forcing_q_hist.append(float(qtarget))

    # Initial fields and snapshot.
    dm = charge_moments(droplet, dr, dq)
    am = charge_moments(aerosol, ar, aq)
    rho = _charge_density(npos, nneg, am, dm)
    jtarget = _forcing_value(cfg.forcing.current_mode, t, cfg.forcing.current_density_initial_a_m2, cfg.forcing.current_density_final_a_m2, cfg.forcing)
    qtarget = _forcing_value(cfg.forcing.ionization_mode, t, cfg.forcing.ionization_initial_m3_s, cfg.forcing.ionization_final_m3_s, cfg.forcing)
    jcond, _current_parts = _non_displacement_current(
        npos, nneg, electric, z, mu_pos, mu_neg, d_pos, d_neg,
        0.0, cfg.transport.ion_eddy_diffusivity_m2_s, cfg.transport.eddy_diffusivity_m2_s,
        am, droplet, dr, dq, cfg.transport.include_droplet_sedimentation,
        cfg.electrodynamics.include_ion_diffusion_current,
    )
    snapshots[0.0] = _snapshot(model, 0.0, droplet, aerosol, dr, dre, dq, ar, are, aq, npos, nneg, electric, rho, jcond, last_jdisp, jtarget)
    next_snapshot = 1
    diagnostics(jtarget, qtarget, jcond, last_jdisp, rho)
    next_diag = cfg.numerics.diagnostics_interval_s
    if progress_callback:
        progress_callback(0.0, "前沿谱模型初始化：开始离子—气溶胶—云滴—电场耦合积分")

    while t < duration - 1.0e-12:
        if cancel_callback and cancel_callback():
            raise SimulationCancelled("Simulation cancelled by user.")
        jtarget = _forcing_value(cfg.forcing.current_mode, t, cfg.forcing.current_density_initial_a_m2, cfg.forcing.current_density_final_a_m2, cfg.forcing)
        qtarget = _forcing_value(cfg.forcing.ionization_mode, t, cfg.forcing.ionization_initial_m3_s, cfg.forcing.ionization_final_m3_s, cfg.forcing)

        dm = charge_moments(droplet, dr, dq)
        am = charge_moments(aerosol, ar, aq)
        rp_d, rn_d = charging_rates(dr, dq, npos, nneg, d_pos, d_neg, temperature, electric, cfg.base.physics.include_drift_charging)
        rp_a, rn_a = charging_rates(ar, aq, npos, nneg, d_pos, d_neg, temperature, electric, cfg.base.physics.include_drift_charging, correction if correction.enabled else None)

        # Backward-Euler event counts are evaluated at the updated spectrum;
        # use a trial update to estimate ion sinks and the positivity-limited dt.
        proposed = min(cfg.numerics.max_dt_s, duration - t)
        future = [x for x in requested + [next_diag, cfg.forcing.step_time_s] if x > t + 1.0e-10]
        if future:
            proposed = min(proposed, min(future) - t)
        max_v = abs(cfg.transport.air_velocity_down_m_s) + float(np.max(terminal_velocity_stokes(dr)))
        proposed = transport_dt_limit(
            dz,
            max_v,
            max(cfg.transport.eddy_diffusivity_m2_s, cfg.transport.ion_eddy_diffusivity_m2_s),
            cfg.numerics.cfl_safety,
            proposed,
        )
        trial_d = implicit_charge_step(droplet, rp_d, rn_d, proposed)
        trial_a = implicit_charge_step(aerosol, rp_a, rn_a, proposed)
        loss_pos = np.sum(rp_d * trial_d, axis=(1, 2)) + np.sum(rp_a * trial_a, axis=(1, 2))
        loss_neg = np.sum(rn_d * trial_d, axis=(1, 2)) + np.sum(rn_a * trial_a, axis=(1, 2))
        recomb = alpha * npos * nneg
        dt = _reaction_dt_limit(npos, nneg, loss_pos, loss_neg, recomb, proposed, cfg.numerics.cfl_safety)
        if dt < cfg.numerics.min_dt_s and proposed > cfg.numerics.min_dt_s:
            dt = cfg.numerics.min_dt_s
        dt = min(dt, proposed)
        if dt <= 0:
            raise RuntimeError("Non-positive frontier time step.")

        # Charge-state kinetics and exactly matched implicit ion capture counts.
        droplet_new = implicit_charge_step(droplet, rp_d, rn_d, dt)
        aerosol_new = implicit_charge_step(aerosol, rp_a, rn_a, dt)
        loss_pos = np.sum(rp_d * droplet_new, axis=(1, 2)) + np.sum(rp_a * aerosol_new, axis=(1, 2))
        loss_neg = np.sum(rn_d * droplet_new, axis=(1, 2)) + np.sum(rn_a * aerosol_new, axis=(1, 2))

        # Ion chemistry.
        recomb = alpha * npos * nneg
        npos = np.maximum(npos + dt * (qtarget - recomb - loss_pos), 1.0)
        nneg = np.maximum(nneg + dt * (qtarget - recomb - loss_neg), 1.0)

        # Vertical transport of ions and particles. z is positive downward.
        if cfg.transport.enabled:
            phase = 2.0 * PI * t / max(cfg.transport.cloud_edge_pulsation_period_s, 1.0)
            air = cfg.transport.air_velocity_down_m_s + cfg.transport.cloud_edge_pulsation_amplitude_m_s * math.sin(phase)
            vpos = air + mu_pos * electric
            vneg = air - mu_neg * electric
            npos = transport_step(
                npos,
                vpos,
                d_pos + cfg.transport.ion_eddy_diffusivity_m2_s,
                dz,
                dt,
                lower_boundary="noflux",
                upper_boundary="open",
            )
            nneg = transport_step(
                nneg,
                vneg,
                d_neg + cfg.transport.ion_eddy_diffusivity_m2_s,
                dz,
                dt,
                lower_boundary="open",
                upper_boundary="noflux",
            )
            aerosol_new = transport_step(
                aerosol_new, air, cfg.transport.eddy_diffusivity_m2_s, dz, dt
            )
            for ir, settling in enumerate(terminal_velocity_stokes(dr) if cfg.transport.include_droplet_sedimentation else np.zeros_like(dr)):
                droplet_new[:, ir, :] = transport_step(
                    droplet_new[:, ir, :],
                    air + settling,
                    cfg.transport.eddy_diffusivity_m2_s,
                    dz,
                    dt,
                    upper_boundary="open",
                )

        # Prescribed condensation/evaporation advection in radius space.
        if cfg.transport.include_condensational_growth:
            supersaturation = cfg.transport.supersaturation_peak * model.cloud_shape
            radius_velocity = (
                cfg.transport.condensational_growth_coefficient_m2_s
                * supersaturation[:, None]
                / np.maximum(dr[None, :], 1.0e-12)
            )
            droplet_new = radius_advection_step(droplet_new, dre, radius_velocity, dt)

        # Optional charge-resolved aerosol scavenging by droplets.  Scavenging
        # evolves much more slowly than ion charging, so it is operator-split
        # on a configurable interval rather than rebuilding and applying the
        # four-dimensional pair kernel at every sub-second charge step.
        if cfg.scavenging.enabled:
            scavenging_elapsed_s += dt
            do_scavenging = (
                scavenging_elapsed_s >= cfg.scavenging.update_interval_s - 1.0e-12
                or t + dt >= duration - 1.0e-12
            )
            if do_scavenging:
                aerosol_new, droplet_new, last_scav = apply_electroscavenging(
                    aerosol_new,
                    droplet_new,
                    ar,
                    aq,
                    dr,
                    dq,
                    temperature,
                    scavenging_elapsed_s,
                    cfg.scavenging.kernel_scale,
                    cfg.scavenging.max_fraction_per_step,
                    cfg.scavenging.active_transfer,
                    cfg.scavenging.include_brownian,
                    cfg.scavenging.include_interception,
                    cfg.scavenging.include_electrostatic,
                    cfg.scavenging.max_droplet_transfer_fraction_per_update,
                    cfg.scavenging.vertical_block_size,
                )
                scavenging_elapsed_s = 0.0
            else:
                last_scav = ScavengingDiagnostics(zero.copy(), zero.copy(), np.ones_like(z))
        else:
            last_scav = ScavengingDiagnostics(zero.copy(), zero.copy(), np.ones_like(z))

        if cfg.electrodynamics.boundary_condition_mode == "clear_air_reservoir":
            nb = min(cfg.electrodynamics.boundary_reservoir_cells, max(1, z.size // 4))
            aerosol_new[:nb, :, :] = aerosol_reservoir_top[None, :, :]
            aerosol_new[-nb:, :, :] = aerosol_reservoir_bottom[None, :, :]
            droplet_new[:nb, :, :] = 0.0
            droplet_new[-nb:, :, :] = 0.0

        droplet = np.maximum(droplet_new, 0.0)
        aerosol = np.maximum(aerosol_new, 0.0)

        # Inflow conditions for ion number fluxes.
        drop_surface = np.sum(charge_moments(droplet, dr, dq)["size_number_m3"] * dr[None, :], axis=1)
        aero_surface = np.sum(charge_moments(aerosol, ar, aq)["size_number_m3"] * ar[None, :], axis=1)
        clear_npos, clear_nneg = _local_ion_equilibrium(
            qtarget,
            alpha,
            d_pos,
            d_neg,
            np.zeros_like(z),
            np.full_like(z, aero_surface[0]),
        )
        if cfg.electrodynamics.boundary_condition_mode == "clear_air_reservoir":
            nb = min(cfg.electrodynamics.boundary_reservoir_cells, max(1, z.size // 4))
            npos[:nb] = clear_npos[:nb]
            nneg[:nb] = clear_nneg[:nb]
            npos[-nb:] = clear_npos[-nb:]
            nneg[-nb:] = clear_nneg[-nb:]
        else:
            npos[0] = clear_npos[0]
            nneg[-1] = clear_nneg[-1]

        # Dynamic Maxwell-current closure followed by a mild Gauss projection.
        dm = charge_moments(droplet, dr, dq)
        am = charge_moments(aerosol, ar, aq)
        rho = _charge_density(npos, nneg, am, dm)
        air_for_current = air if cfg.transport.enabled else 0.0
        jcond, current_parts = _non_displacement_current(
            npos, nneg, electric, z, mu_pos, mu_neg, d_pos, d_neg,
            air_for_current, cfg.transport.ion_eddy_diffusivity_m2_s,
            cfg.transport.eddy_diffusivity_m2_s, am, droplet, dr, dq,
            cfg.transport.include_droplet_sedimentation,
            cfg.electrodynamics.include_ion_diffusion_current,
        )
        previous_electric = electric.copy()
        sigma_current = np.maximum(current_parts["sigma_current_s_m"], 1.0e-30)
        diffusion_current = current_parts["other_current_a_m2"]
        if cfg.electrodynamics.include_displacement_current:
            if step % cfg.electrodynamics.projection_interval_steps == 0:
                electric = _solve_field_variational(
                    electric,
                    rho,
                    sigma_current,
                    diffusion_current,
                    jtarget,
                    dt,
                    dz,
                    cfg.electrodynamics.gauss_projection_relaxation,
                    (
                        jtarget
                        / (ELEMENTARY_CHARGE * (mu_pos * clear_npos[0] + mu_neg * clear_nneg[0]))
                        if cfg.electrodynamics.boundary_condition_mode == "clear_air_reservoir"
                        else None
                    ),
                    cfg.electrodynamics.boundary_field_anchor_weight,
                )
            else:
                electric = (
                    electric + dt * (jtarget - diffusion_current) / EPSILON_0
                ) / (1.0 + dt * sigma_current / EPSILON_0)
        else:
            electric = (jtarget - diffusion_current) / sigma_current
        electric = np.clip(
            electric,
            cfg.electrodynamics.electric_field_floor_v_m,
            cfg.electrodynamics.electric_field_ceiling_v_m,
        )
        last_jdisp = EPSILON_0 * (electric - previous_electric) / dt
        jcond, current_parts = _non_displacement_current(
            npos, nneg, electric, z, mu_pos, mu_neg, d_pos, d_neg,
            air_for_current, cfg.transport.ion_eddy_diffusivity_m2_s,
            cfg.transport.eddy_diffusivity_m2_s, am, droplet, dr, dq,
            cfg.transport.include_droplet_sedimentation,
            cfg.electrodynamics.include_ion_diffusion_current,
        )

        t += dt
        step += 1
        while next_snapshot < len(requested) and t >= requested[next_snapshot] - 1.0e-8:
            st = requested[next_snapshot]
            snapshots[st] = _snapshot(
                model,
                st,
                droplet,
                aerosol,
                dr,
                dre,
                dq,
                ar,
                are,
                aq,
                npos,
                nneg,
                electric,
                rho,
                jcond,
                last_jdisp,
                jtarget,
            )
            next_snapshot += 1
        if t >= next_diag - 1.0e-8:
            diagnostics(jtarget, qtarget, jcond, last_jdisp, rho)
            next_diag += cfg.numerics.diagnostics_interval_s
        if progress_callback and (step == 1 or step % 20 == 0):
            dm_now = charge_moments(droplet, dr, dq)
            progress_callback(
                min(1.0, t / duration),
                f"t={t:.1f}s, Jz={jtarget/1e-12:.2f} pA m⁻², "
                f"云顶/云底电荷={np.max(dm_now['mean_charge_e'][top_mask]):+.1f}/"
                f"{np.min(dm_now['mean_charge_e'][base_mask]):+.1f} e",
            )

    # Final diagnostics and metadata.
    dm = charge_moments(droplet, dr, dq)
    am = charge_moments(aerosol, ar, aq)
    rho = _charge_density(npos, nneg, am, dm)
    jtarget = _forcing_value(cfg.forcing.current_mode, t, cfg.forcing.current_density_initial_a_m2, cfg.forcing.current_density_final_a_m2, cfg.forcing)
    qtarget = _forcing_value(cfg.forcing.ionization_mode, t, cfg.forcing.ionization_initial_m3_s, cfg.forcing.ionization_final_m3_s, cfg.forcing)
    final_air = cfg.transport.air_velocity_down_m_s + (
        cfg.transport.cloud_edge_pulsation_amplitude_m_s
        * math.sin(2.0 * PI * t / max(cfg.transport.cloud_edge_pulsation_period_s, 1.0))
        if cfg.transport.enabled else 0.0
    )
    jcond, _current_parts = _non_displacement_current(
        npos, nneg, electric, z, mu_pos, mu_neg, d_pos, d_neg,
        final_air, cfg.transport.ion_eddy_diffusivity_m2_s, cfg.transport.eddy_diffusivity_m2_s,
        am, droplet, dr, dq, cfg.transport.include_droplet_sedimentation,
        cfg.electrodynamics.include_ion_diffusion_current,
    )
    if not diag_time or abs(diag_time[-1] - t) > 1.0e-8:
        diagnostics(jtarget, qtarget, jcond, last_jdisp, rho)
    snapshots[duration] = _snapshot(
        model,
        duration,
        droplet,
        aerosol,
        dr,
        dre,
        dq,
        ar,
        are,
        aq,
        npos,
        nneg,
        electric,
        rho,
        jcond,
        last_jdisp,
        jtarget,
    )
    gauss = np.gradient(electric, z, edge_order=2) - rho / EPSILON_0
    total_current = jcond + last_jdisp
    upper_column = float(np.trapezoid(rho[top_mask], z[top_mask]))
    lower_column = float(np.trapezoid(rho[base_mask], z[base_mask]))
    upper_indices = np.where(top_mask)[0]
    lower_indices = np.where(base_mask)[0]
    upper_rho_index = int(upper_indices[np.argmax(rho[top_mask])])
    lower_rho_index = int(lower_indices[np.argmin(rho[base_mask])])
    upper_rho_peak = float(rho[upper_rho_index])
    lower_rho_peak = float(rho[lower_rho_index])
    upper_boundary_ratio = float(abs(rho[0]) / max(abs(upper_rho_peak), 1.0e-300))
    lower_boundary_ratio = float(abs(rho[-1]) / max(abs(lower_rho_peak), 1.0e-300))
    upper_threshold = 0.01 * abs(upper_rho_peak)
    lower_threshold = 0.01 * abs(lower_rho_peak)
    upper_tail_candidates = np.where((z < z[upper_rho_index]) & (np.abs(rho) < upper_threshold))[0]
    lower_tail_candidates = np.where((z > z[lower_rho_index]) & (np.abs(rho) < lower_threshold))[0]
    upper_tail_z = float(z[upper_tail_candidates[-1]]) if upper_tail_candidates.size else float('nan')
    lower_tail_z = float(z[lower_tail_candidates[0]]) if lower_tail_candidates.size else float('nan')
    droplet_number = dm['number_m3']
    droplet_mask = droplet_number > 0.01 * max(float(np.max(droplet_number)), 1.0e-300)
    droplet_extent_low = float(z[np.where(droplet_mask)[0][0]]) if np.any(droplet_mask) else float('nan')
    droplet_extent_high = float(z[np.where(droplet_mask)[0][-1]]) if np.any(droplet_mask) else float('nan')
    diagnostics_dict: dict[str, np.ndarray | float | str | bool] = {
        "solver_mode": "frontier_multispectrum_maxwell",
        "time_s": np.asarray(diag_time),
        "top_droplet_charge_e": np.asarray(diag_top_q),
        "bottom_droplet_charge_e": np.asarray(diag_base_q),
        "top_aerosol_charge_e": np.asarray(diag_top_aq),
        "bottom_aerosol_charge_e": np.asarray(diag_base_aq),
        "max_electric_field_v_m": np.asarray(diag_max_e),
        "max_gauss_residual_v_m2": np.asarray(diag_gauss),
        "max_relative_gauss_residual": np.asarray(diag_gauss_relative),
        "max_boundary_gauss_residual_v_m2": np.asarray(diag_gauss_boundary),
        "max_relative_total_current_residual": np.asarray(diag_current),
        "column_charge_c_m2": np.asarray(diag_total_charge),
        "mean_displacement_current_fraction": np.asarray(diag_displacement_fraction),
        "column_aerosol_scavenged_m2": np.asarray(diag_scavenged),
        "mean_electrostatic_scavenging_multiplier": np.asarray(diag_mult),
        "scheduled_current_density_a_m2": np.asarray(forcing_j_hist),
        "scheduled_ion_pair_production_m3_s": np.asarray(forcing_q_hist),
        "upper_peak_droplet_charge_e": float(np.max(dm["mean_charge_e"][top_mask])),
        "lower_peak_droplet_charge_e": float(np.min(dm["mean_charge_e"][base_mask])),
        "upper_integrated_space_charge_c_m2": upper_column,
        "lower_integrated_space_charge_c_m2": lower_column,
        "upper_space_charge_peak_c_m3": upper_rho_peak,
        "lower_space_charge_peak_c_m3": lower_rho_peak,
        "upper_space_charge_peak_z_m": float(z[upper_rho_index]),
        "lower_space_charge_peak_z_m": float(z[lower_rho_index]),
        "upper_boundary_to_peak_charge_ratio": upper_boundary_ratio,
        "lower_boundary_to_peak_charge_ratio": lower_boundary_ratio,
        "upper_one_percent_tail_z_m": upper_tail_z,
        "lower_one_percent_tail_z_m": lower_tail_z,
        "droplet_one_percent_extent_low_z_m": droplet_extent_low,
        "droplet_one_percent_extent_high_z_m": droplet_extent_high,
        "opposite_droplet_sign_test": bool(np.max(dm["mean_charge_e"][top_mask]) > 0 and np.min(dm["mean_charge_e"][base_mask]) < 0),
        "opposite_space_charge_sign_test": bool(upper_column > 0 and lower_column < 0),
        "final_max_gauss_residual_v_m2": float(np.max(np.abs(gauss))),
        "final_max_relative_gauss_residual": float(
            np.max(np.abs(gauss[2:-2] if z.size > 5 else gauss))
            / max(float(np.max(np.abs((rho / EPSILON_0)[2:-2] if z.size > 5 else rho / EPSILON_0))), 1.0e-3)
        ),
        "final_max_boundary_gauss_residual_v_m2": float(
            np.max(np.abs(np.concatenate((gauss[:2], gauss[-2:])))) if z.size > 5 else np.max(np.abs(gauss))
        ),
        "final_max_relative_total_current_residual": float(np.max(np.abs(total_current - jtarget)) / max(abs(jtarget), 1.0e-30)),
        "time_steps": float(step),
    }
    metadata = {
        "config": cfg.to_dict(),
        "atmosphere": asdict(model.atmosphere),
        "state_definition": {
            "droplets": "N_d(r,k,z,t), bin-integrated m^-3",
            "aerosols": "N_a(a,k,z,t), bin-integrated m^-3",
            "ions": "n_+(z,t), n_-(z,t)",
            "electric_field": "E_down(z,t), positive downward",
        },
        "innovations": [
            "finite displacement current in total-current closure",
            "charge-resolved aerosol and droplet spectra",
            "vertical turbulent transport and sedimentation",
            "optional radius-space condensation advection",
            "optional charge-resolved electroscavenging",
            "observation and ensemble-inversion ready output",
        ],
        "attachment_measurement_table": cfg.aerosol_spectrum.measured_attachment_table or "theory_only",
        "model_limits": (
            "One-dimensional warm-layer-cloud research model. Turbulence is represented by eddy diffusion; "
            "the electroscavenging kernel is a compact hypothesis-testing kernel and should be calibrated "
            "against trajectory calculations or laboratory data before quantitative cloud-impact claims."
        ),
    }
    if progress_callback:
        progress_callback(1.0, "前沿云层电荷谱模拟完成")
    return SimulationResult(
        z_m=z.copy(),
        snapshot_times_s=np.asarray(sorted(snapshots.keys()), dtype=float),
        snapshots=snapshots,
        diagnostics=diagnostics_dict,
        metadata=metadata,
    )
