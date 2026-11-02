from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import erf

from .atmosphere import AtmosphereState
from .constants import (
    ATOMIC_MASS_UNIT,
    BOLTZMANN,
    ELEMENTARY_CHARGE,
    EPSILON_0,
    GAS_CONSTANT,
    PI,
)


@dataclass(frozen=True)
class AttachmentCoefficients:
    beta_positive_m3_s: float
    beta_negative_m3_s: float
    gamma_positive_m3_s: float
    gamma_negative_m3_s: float
    metadata: dict[str, float | str]


def _thermal_velocities(temperature_k: float) -> tuple[float, float, float]:
    carrier = math.sqrt(8.0 * temperature_k * GAS_CONSTANT * 1.0e3 / 29.5 / PI)
    positive = math.sqrt(8.0 * temperature_k * GAS_CONSTANT * 1.0e3 / 150.0 / PI)
    negative = math.sqrt(8.0 * temperature_k * GAS_CONSTANT * 1.0e3 / 80.0 / PI)
    return carrier, positive, negative


def _mean_free_path(
    carrier_speed: float,
    mobility: float,
    ion_molar_mass: float,
) -> float:
    return (
        (4.0 / 3.0)
        * (29.5 * ATOMIC_MASS_UNIT * carrier_speed * mobility / ELEMENTARY_CHARGE)
        * math.sqrt(ion_molar_mass / (29.5 + ion_molar_mass))
    )


def _contact_delta_charged(tm: float, charge_state: float = -1.0) -> float:
    p = charge_state

    def objective(x: float) -> float:
        return (
            x**5
            - p * tm * x**4
            - 2.0 * x**3
            + 2.0 * p * tm * x**2
            + (1.0 - tm) * x
            - p * tm
        )

    lo, hi = 1.0, 300.0
    f_lo, f_hi = objective(lo), objective(hi)
    if f_lo == 0.0:
        return lo
    if f_hi == 0.0:
        return hi
    if f_lo * f_hi > 0.0:
        # Search a logarithmic bracket rather than allowing an endless IDL while-loop.
        xs = np.geomspace(1.0, 1.0e5, 800)
        fs = [objective(float(x)) for x in xs]
        for x1, x2, y1, y2 in zip(xs[:-1], xs[1:], fs[:-1], fs[1:]):
            if y1 == 0.0:
                return float(x1)
            if y1 * y2 < 0.0:
                lo, hi = float(x1), float(x2)
                break
        else:
            raise RuntimeError(
                "Unable to bracket the Hoppel–Keefe charged-particle contact root."
            )
    return float(brentq(objective, lo, hi, xtol=1.0e-12, rtol=1.0e-12, maxiter=200))


def _legacy_shifted_boole_integral(
    func: Callable[[np.ndarray], np.ndarray], start: float, stop: float, dx: float = 0.01
) -> float:
    """Reproduce the uploaded program's shifted five-point Boole-like summation.

    The original uses nodes x1 + (ix+1) * dx/4, so the fifth node lies beyond x2.
    This mode is retained only for numerical reproduction and comparison.
    """
    if start >= stop:
        return 0.0
    total = 0.0
    x1 = float(start)
    weights = np.array([7.0, 32.0, 12.0, 32.0, 7.0])
    while x1 < stop:
        x2 = min(x1 + dx, stop)
        ga = (x2 - x1) / 4.0
        nodes = x1 + ga * np.arange(1.0, 6.0)
        vals = func(nodes)
        total += (x2 - x1) / 90.0 * float(np.dot(weights, vals))
        x1 = x2
    return total


def _corrected_integral(func: Callable[[float], float], start: float, stop: float) -> float:
    if start >= stop:
        return 0.0
    value, _ = quad(func, start, stop, epsabs=1.0e-12, epsrel=2.0e-9, limit=300)
    return float(value)


def _coefficient_for_ion(
    temperature_k: float,
    aerosol_radius_m: float,
    ion_mobility: float,
    ion_diffusivity: float,
    ion_speed: float,
    ion_mean_free_path: float,
    mode: str,
) -> tuple[float, float, dict[str, float]]:
    r = aerosol_radius_m
    beta = math.nan
    gamma = math.nan
    details: dict[str, float] = {}

    for charge_state in (0.0, -1.0):
        p = charge_state
        if p == 0.0:
            tm = (
                ELEMENTARY_CHARGE**2 * PI
                / (r * 8.0 * BOLTZMANN * temperature_k * 4.0 * PI * EPSILON_0)
            )
            delta_dimensionless = math.sqrt(1.0 + math.sqrt(tm))
        else:
            tm = ELEMENTARY_CHARGE**2 / (
                r * BOLTZMANN * temperature_k * 4.0 * PI * EPSILON_0
            )
            delta_dimensionless = _contact_delta_charged(tm, p)

        delta = delta_dimensionless * r

        if p == 0.0:
            enhancement = 1.0 + math.sqrt(
                ELEMENTARY_CHARGE**2
                * PI
                / (2.0 * r * BOLTZMANN * temperature_k * 4.0 * PI * EPSILON_0)
            )
        else:
            py = ELEMENTARY_CHARGE**2 / (
                2.0 * r * BOLTZMANN * temperature_k * 4.0 * PI * EPSILON_0
            )
            if mode == "legacy_shifted_boole":
                singular_integral = _legacy_shifted_boole_integral(
                    lambda x: np.exp(-x) * np.power(x, -0.5), 0.0, 20.0
                )
            else:
                # Integral_0^20 exp(-x) x^-1/2 dx = sqrt(pi) erf(sqrt(20)).
                singular_integral = math.sqrt(PI) * float(erf(math.sqrt(20.0)))
            enhancement = (
                1.0
                + math.exp(py) * math.sqrt(py) * singular_integral
                + 15.0 * py / 4.0
            )

        lower = (delta + ion_mean_free_path) / r

        def potential_integrand_scalar(x: float) -> float:
            denominator = 2.0 * r**2 * x**2 * (r**2 * x**2 - r**2)
            potential = (
                p * ELEMENTARY_CHARGE**2 / (x * r)
                - ELEMENTARY_CHARGE**2 * r**3 / denominator
            ) / (4.0 * PI * EPSILON_0 * BOLTZMANN * temperature_k)
            potential = min(700.0, max(-700.0, potential))
            return math.exp(potential) / x**2

        if mode == "legacy_shifted_boole":
            def vector_integrand(x: np.ndarray) -> np.ndarray:
                denominator = 2.0 * r**2 * x**2 * (r**2 * x**2 - r**2)
                potential = (
                    p * ELEMENTARY_CHARGE**2 / (x * r)
                    - ELEMENTARY_CHARGE**2 * r**3 / denominator
                ) / (4.0 * PI * EPSILON_0 * BOLTZMANN * temperature_k)
                return np.exp(np.clip(potential, -700.0, 700.0)) / x**2

            cy = _legacy_shifted_boole_integral(vector_integrand, lower, 20.0)
        else:
            cy = _corrected_integral(potential_integrand_scalar, lower, 20.0)

        fd = PI * r**2 * ion_speed * enhancement
        distance = delta + ion_mean_free_path
        g1 = (
            p * ELEMENTARY_CHARGE**2 / distance
            - ELEMENTARY_CHARGE**2
            * r**3
            / (2.0 * distance**2 * (distance**2 - r**2))
        ) / (4.0 * PI * EPSILON_0 * BOLTZMANN * temperature_k)

        denominator = math.exp(min(700.0, g1)) + fd / (
            4.0 * PI * ion_diffusivity * r
        ) * cy
        coefficient = fd / denominator
        if p == 0.0:
            beta = coefficient
            details["neutral_delta_m"] = delta
            details["neutral_enhancement"] = enhancement
        else:
            gamma = coefficient
            details["charged_delta_m"] = delta
            details["charged_enhancement"] = enhancement

    return beta, gamma, details



def _paper2007_table_coefficients(
    altitude_km: float, aerosol_radius_m: float
) -> AttachmentCoefficients:
    """Log-bilinear interpolation of Table 1 in Zhou and Tinsley (2007).

    Tabulated values are for altitudes 2, 4, and 10 km and radii 0.02,
    0.04, and 0.10 micrometres. Values outside the table are clamped to
    the nearest tabulated altitude/radius to avoid unsupported extrapolation.
    """
    altitudes = np.array([2.0, 4.0, 10.0], dtype=float)
    radii_m = np.array([2.0e-8, 4.0e-8, 1.0e-7], dtype=float)
    # axis order: altitude, radius; values in m3 s-1
    beta_pos = 1.0e-12 * np.array([
        [0.61, 1.53, 4.55],
        [0.63, 1.61, 4.90],
        [0.70, 1.93, 6.62],
    ])
    beta_neg = 1.0e-12 * np.array([
        [0.86, 2.16, 6.38],
        [0.88, 2.27, 6.88],
        [0.96, 2.70, 9.34],
    ])
    gamma_pos = 1.0e-12 * np.array([
        [2.36, 3.32, 6.23],
        [2.56, 3.64, 6.83],
        [3.06, 4.84, 9.72],
    ])
    gamma_neg = 1.0e-12 * np.array([
        [3.13, 4.55, 8.65],
        [3.37, 4.98, 9.49],
        [3.91, 6.49, 13.47],
    ])

    h = float(np.clip(altitude_km, altitudes[0], altitudes[-1]))
    r = float(np.clip(aerosol_radius_m, radii_m[0], radii_m[-1]))

    def interp(table: np.ndarray) -> float:
        # First interpolate in log(radius), then in altitude. Coefficients are
        # positive and vary smoothly, so log interpolation is appropriate.
        log_r = np.log(radii_m)
        at_h = np.array([
            np.exp(np.interp(np.log(r), log_r, np.log(row))) for row in table
        ])
        return float(np.exp(np.interp(h, altitudes, np.log(at_h))))

    return AttachmentCoefficients(
        beta_positive_m3_s=interp(beta_pos),
        beta_negative_m3_s=interp(beta_neg),
        gamma_positive_m3_s=interp(gamma_pos),
        gamma_negative_m3_s=interp(gamma_neg),
        metadata={
            "mode": "paper2007_table",
            "source": "Zhou and Tinsley (2007), Table 1",
            "requested_altitude_km": float(altitude_km),
            "used_altitude_km": h,
            "requested_aerosol_radius_m": float(aerosol_radius_m),
            "used_aerosol_radius_m": r,
            "table_clamped": bool(h != altitude_km or r != aerosol_radius_m),
        },
    )

def calculate_attachment_coefficients(
    atmosphere: AtmosphereState,
    aerosol_radius_m: float,
    mode: str = "corrected",
) -> AttachmentCoefficients:
    if mode == "paper2007_table":
        return _paper2007_table_coefficients(atmosphere.altitude_km, aerosol_radius_m)
    if mode not in {"corrected", "legacy_shifted_boole"}:
        raise ValueError(f"Unknown attachment mode: {mode}")

    carrier_speed, positive_speed, negative_speed = _thermal_velocities(
        atmosphere.temperature_k
    )
    positive_mfp = _mean_free_path(
        carrier_speed, atmosphere.positive_mobility_m2_v_s, 150.0
    )
    negative_mfp = _mean_free_path(
        carrier_speed, atmosphere.negative_mobility_m2_v_s, 80.0
    )

    beta_pos, gamma_pos, pos_details = _coefficient_for_ion(
        atmosphere.temperature_k,
        aerosol_radius_m,
        atmosphere.positive_mobility_m2_v_s,
        atmosphere.positive_diffusivity_m2_s,
        positive_speed,
        positive_mfp,
        mode,
    )
    beta_neg, gamma_neg, neg_details = _coefficient_for_ion(
        atmosphere.temperature_k,
        aerosol_radius_m,
        atmosphere.negative_mobility_m2_v_s,
        atmosphere.negative_diffusivity_m2_s,
        negative_speed,
        negative_mfp,
        mode,
    )

    metadata: dict[str, float | str] = {
        "mode": mode,
        "positive_mean_free_path_m": positive_mfp,
        "negative_mean_free_path_m": negative_mfp,
        **{f"positive_{k}": v for k, v in pos_details.items()},
        **{f"negative_{k}": v for k, v in neg_details.items()},
    }
    return AttachmentCoefficients(
        beta_positive_m3_s=beta_pos,
        beta_negative_m3_s=beta_neg,
        gamma_positive_m3_s=gamma_pos,
        gamma_negative_m3_s=gamma_neg,
        metadata=metadata,
    )
