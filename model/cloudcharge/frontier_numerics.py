from __future__ import annotations

import math
from typing import Callable

import numpy as np
from scipy.special import ndtr

from .constants import BOLTZMANN, ELEMENTARY_CHARGE, EPSILON_0, PI

try:
    from numba import njit, prange
    _NUMBA_AVAILABLE = True
except Exception:
    _NUMBA_AVAILABLE = False

if _NUMBA_AVAILABLE:
    @njit(parallel=True, cache=True)
    def _implicit_charge_step_numba(rhs, rp, rn, dt):
        batch, nq = rhs.shape
        out = np.empty_like(rhs)
        cp = np.empty_like(rhs)
        dp = np.empty_like(rhs)
        for b in prange(batch):
            main0 = 1.0 + dt * (rp[b, 0] + rn[b, 0])
            if main0 < 1.0e-300:
                main0 = 1.0e-300
            cp[b, 0] = (-dt * rn[b, 1] / main0) if nq > 1 else 0.0
            dp[b, 0] = rhs[b, 0] / main0
            for k in range(1, nq):
                lower = -dt * rp[b, k - 1]
                main = 1.0 + dt * (rp[b, k] + rn[b, k])
                denom = main - lower * cp[b, k - 1]
                if denom < 1.0e-300:
                    denom = 1.0e-300
                cp[b, k] = (-dt * rn[b, k + 1] / denom) if k < nq - 1 else 0.0
                dp[b, k] = (rhs[b, k] - lower * dp[b, k - 1]) / denom
            out[b, nq - 1] = dp[b, nq - 1]
            for k in range(nq - 2, -1, -1):
                out[b, k] = dp[b, k] - cp[b, k] * out[b, k + 1]
                if out[b, k] < 0.0:
                    out[b, k] = 0.0
            if out[b, nq - 1] < 0.0:
                out[b, nq - 1] = 0.0
        return out


def bernoulli(x: np.ndarray) -> np.ndarray:
    """Stable B(x)=x/(exp(x)-1)."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    small = np.abs(x) < 1.0e-5
    xs = x[small]
    out[small] = 1.0 - xs / 2.0 + xs**2 / 12.0 - xs**4 / 720.0
    large_pos = x > 50.0
    large_neg = x < -50.0
    regular = ~(small | large_pos | large_neg)
    out[large_pos] = x[large_pos] * np.exp(-x[large_pos])
    out[large_neg] = -x[large_neg]
    out[regular] = x[regular] / np.expm1(x[regular])
    return out


def lognormal_bin_weights(edges: np.ndarray, geometric_mean: float, geometric_std: float) -> np.ndarray:
    sigma = math.log(geometric_std)
    hi = (np.log(edges[1:]) - math.log(geometric_mean)) / sigma
    lo = (np.log(edges[:-1]) - math.log(geometric_mean)) / sigma
    w = ndtr(hi) - ndtr(lo)
    if np.sum(w) <= 0:
        raise ValueError("Lognormal mean lies outside radius grid.")
    return w / np.sum(w)


def gaussian_charge_weights(charges_e: np.ndarray, std_e: float, neutral: bool = False) -> np.ndarray:
    if neutral:
        w = np.zeros(charges_e.size, dtype=float)
        w[int(np.argmin(np.abs(charges_e)))] = 1.0
        return w
    w = np.exp(-0.5 * (charges_e / std_e) ** 2)
    return w / np.sum(w)


def build_radius_grid(rmin: float, rmax: float, bins: int, spacing: str = "log") -> tuple[np.ndarray, np.ndarray]:
    if spacing == "log":
        edges = np.geomspace(rmin, rmax, bins + 1)
        centers = np.sqrt(edges[:-1] * edges[1:])
    else:
        edges = np.linspace(rmin, rmax, bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
    return centers.astype(float), edges.astype(float)


def charge_moments(spectrum: np.ndarray, radius_m: np.ndarray, charges_e: np.ndarray) -> dict[str, np.ndarray]:
    number = np.sum(spectrum, axis=(1, 2))
    safe = np.maximum(number, 1.0e-300)
    size_number = np.sum(spectrum, axis=2)
    charge_number = np.sum(spectrum, axis=1)
    q1 = np.sum(charge_number * charges_e[None, :], axis=1)
    q2 = np.sum(charge_number * charges_e[None, :] ** 2, axis=1)
    r1 = np.sum(size_number * radius_m[None, :], axis=1)
    r2 = np.sum(size_number * radius_m[None, :] ** 2, axis=1)
    mean_q = q1 / safe
    mean_r = r1 / safe
    return {
        "number_m3": number,
        "size_number_m3": size_number,
        "charge_number_m3": charge_number,
        "mean_charge_e": mean_q,
        "charge_std_e": np.sqrt(np.maximum(q2 / safe - mean_q**2, 0.0)),
        "mean_radius_m": mean_r,
        "radius_std_m": np.sqrt(np.maximum(r2 / safe - mean_r**2, 0.0)),
        "charge_number_density_e_m3": q1,
        "positive_fraction": np.sum(charge_number[:, charges_e > 0], axis=1) / safe,
        "negative_fraction": np.sum(charge_number[:, charges_e < 0], axis=1) / safe,
        "neutral_fraction": np.sum(charge_number[:, charges_e == 0], axis=1) / safe,
    }


def charging_rates(
    radii_m: np.ndarray,
    charges_e: np.ndarray,
    n_pos: np.ndarray,
    n_neg: np.ndarray,
    d_pos: float,
    d_neg: float,
    temperature_k: float,
    electric_field_v_m: np.ndarray,
    include_drift: bool,
    measured_multiplier: Callable[[np.ndarray, np.ndarray, str], np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    x = (
        charges_e[None, None, :]
        * ELEMENTARY_CHARGE**2
        / (4.0 * PI * EPSILON_0 * radii_m[None, :, None] * BOLTZMANN * temperature_k)
    )
    fp = bernoulli(x)
    fn = bernoulli(-x)
    drift = np.ones((n_pos.size, radii_m.size, 1), dtype=float)
    if include_drift:
        drift += (
            ELEMENTARY_CHARGE
            * np.abs(electric_field_v_m[:, None, None])
            * radii_m[None, :, None]
            / (2.0 * BOLTZMANN * temperature_k)
        )
    rp = 4.0 * PI * radii_m[None, :, None] * d_pos * n_pos[:, None, None] * fp * drift
    rn = 4.0 * PI * radii_m[None, :, None] * d_neg * n_neg[:, None, None] * fn * drift
    if measured_multiplier is not None:
        rp *= measured_multiplier(radii_m[None, :, None], charges_e[None, None, :], "positive")
        rn *= measured_multiplier(radii_m[None, :, None], charges_e[None, None, :], "negative")
    rp[:, :, -1] = 0.0
    rn[:, :, 0] = 0.0
    return np.maximum(rp, 0.0), np.maximum(rn, 0.0)


def implicit_charge_step(
    spectrum: np.ndarray, rate_pos: np.ndarray, rate_neg: np.ndarray, dt: float
) -> np.ndarray:
    """Backward-Euler birth-death update for all (z,r) columns."""
    nz, nr, nq = spectrum.shape
    batch = nz * nr
    rhs = np.ascontiguousarray(spectrum.reshape(batch, nq))
    rp = np.ascontiguousarray(rate_pos.reshape(batch, nq))
    rn = np.ascontiguousarray(rate_neg.reshape(batch, nq))
    if _NUMBA_AVAILABLE:
        x = _implicit_charge_step_numba(rhs, rp, rn, dt)
    else:
        main = 1.0 + dt * (rp + rn)
        lower = np.zeros_like(main)
        upper = np.zeros_like(main)
        lower[:, 1:] = -dt * rp[:, :-1]
        upper[:, :-1] = -dt * rn[:, 1:]
        cp = np.zeros_like(main)
        dp = np.zeros_like(rhs)
        denom = np.maximum(main[:, 0], 1.0e-300)
        cp[:, 0] = upper[:, 0] / denom
        dp[:, 0] = rhs[:, 0] / denom
        for k in range(1, nq):
            denom = np.maximum(main[:, k] - lower[:, k] * cp[:, k - 1], 1.0e-300)
            if k < nq - 1:
                cp[:, k] = upper[:, k] / denom
            dp[:, k] = (rhs[:, k] - lower[:, k] * dp[:, k - 1]) / denom
        x = np.empty_like(rhs)
        x[:, -1] = dp[:, -1]
        for k in range(nq - 2, -1, -1):
            x[:, k] = dp[:, k] - cp[:, k] * x[:, k + 1]
    out = np.maximum(x.reshape(nz, nr, nq), 0.0)
    old = np.sum(spectrum, axis=2)
    new = np.sum(out, axis=2)
    mask = old > 0
    scale = np.ones_like(old)
    scale[mask] = old[mask] / np.maximum(new[mask], 1.0e-300)
    return out * scale[:, :, None]


def terminal_velocity_stokes(radius_m: np.ndarray, air_viscosity_pa_s: float = 1.7e-5) -> np.ndarray:
    rho_w = 1000.0
    rho_air = 0.8
    g = 9.80665
    v = 2.0 * (rho_w - rho_air) * g * radius_m**2 / (9.0 * air_viscosity_pa_s)
    # Smoothly cap because Stokes law is inaccurate for large drops.
    return np.minimum(v, 9.0)


def _flux_divergence_scalar(
    values: np.ndarray,
    velocity_faces: np.ndarray,
    diffusivity: float,
    dz: float,
    lower_boundary: str = "noflux",
    upper_boundary: str = "noflux",
) -> np.ndarray:
    """Conservative 1-D advection-diffusion divergence for z increasing downward."""
    n = values.shape[0]
    flux = np.zeros((n + 1,) + values.shape[1:], dtype=float)
    # Interior upwind advection.
    vf = velocity_faces[1:n]
    left = values[:-1]
    right = values[1:]
    upwind = np.where(vf.reshape((-1,) + (1,) * (values.ndim - 1)) >= 0.0, left, right)
    flux[1:n] = vf.reshape((-1,) + (1,) * (values.ndim - 1)) * upwind
    if diffusivity > 0:
        flux[1:n] -= diffusivity * (right - left) / dz
    if lower_boundary == "open":
        flux[0] = velocity_faces[0] * values[0]
    if upper_boundary == "open":
        flux[-1] = velocity_faces[-1] * values[-1]
    return -(flux[1:] - flux[:-1]) / dz


def transport_step(
    values: np.ndarray,
    velocity_center: np.ndarray | float,
    diffusivity: float,
    dz: float,
    dt: float,
    lower_boundary: str = "noflux",
    upper_boundary: str = "noflux",
) -> np.ndarray:
    if np.isscalar(velocity_center):
        vc = np.full(values.shape[0], float(velocity_center), dtype=float)
    else:
        vc = np.asarray(velocity_center, dtype=float)
    vf = np.empty(vc.size + 1, dtype=float)
    vf[1:-1] = 0.5 * (vc[:-1] + vc[1:])
    vf[0] = vc[0]
    vf[-1] = vc[-1]
    tendency = _flux_divergence_scalar(
        values, vf, diffusivity, dz, lower_boundary=lower_boundary, upper_boundary=upper_boundary
    )
    return np.maximum(values + dt * tendency, 0.0)


def transport_dt_limit(
    dz: float,
    max_velocity: float,
    diffusivity: float,
    safety: float,
    proposed: float,
) -> float:
    adv = math.inf if max_velocity <= 0 else safety * dz / max_velocity
    diff = math.inf if diffusivity <= 0 else safety * 0.5 * dz * dz / diffusivity
    return max(min(proposed, adv, diff), 1.0e-8)


def radius_advection_step(
    spectrum: np.ndarray,
    radius_edges_m: np.ndarray,
    radius_velocity_m_s: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Conservative upwind advection in radius space.

    radius_velocity_m_s may have shape (nz,nr) or (nr,). Charge is retained
    while number moves between adjacent radius bins.
    """
    nz, nr, nq = spectrum.shape
    widths = np.diff(radius_edges_m)
    if radius_velocity_m_s.ndim == 1:
        vel = np.broadcast_to(radius_velocity_m_s[None, :], (nz, nr))
    else:
        vel = radius_velocity_m_s
    face_vel = np.zeros((nz, nr + 1), dtype=float)
    face_vel[:, 1:-1] = 0.5 * (vel[:, :-1] + vel[:, 1:])
    face_vel[:, 0] = vel[:, 0]
    face_vel[:, -1] = vel[:, -1]
    density = spectrum / widths[None, :, None]
    flux = np.zeros((nz, nr + 1, nq), dtype=float)
    for i in range(1, nr):
        v = face_vel[:, i]
        flux[:, i, :] = np.where(
            v[:, None] >= 0, v[:, None] * density[:, i - 1, :], v[:, None] * density[:, i, :]
        )
    # Reflecting radius boundaries; spectrum is bin-integrated number.
    tendency = -(flux[:, 1:, :] - flux[:, :-1, :])
    return np.maximum(spectrum + dt * tendency, 0.0)
