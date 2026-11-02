from __future__ import annotations

from dataclasses import dataclass
import math

from .config import AtmosphereConfig
from .constants import BOLTZMANN, ELEMENTARY_CHARGE


@dataclass(frozen=True)
class AtmosphereState:
    altitude_km: float
    temperature_k: float
    pressure_hpa: float
    neutral_molecular_concentration_cm3: float
    mobility_scale: float
    positive_mobility_m2_v_s: float
    negative_mobility_m2_v_s: float
    positive_diffusivity_m2_s: float
    negative_diffusivity_m2_s: float


def temperature_profile(altitude_km: float) -> float:
    h = altitude_km
    if h < 5.0:
        return 288.0 - ((288.0 - 219.0) / 10.0) * h
    if h < 13.0:
        return 254.19 - (35.19 / 8.1) * (h - 4.9)
    if h < 20.0:
        return 219.0
    return 219.0 + (h - 20.0) * 2.0


def pressure_profile_hpa(altitude_km: float) -> float:
    h = altitude_km
    ds1, ds2, ds3 = 9.2080, 6.5059, 6.3779
    a1, a2, a3 = 2.547, 0.8597, 0.1849
    p0, p1, p2 = 1013.2, 280.03, 73.12
    if h < 10.0:
        return (math.exp(-h / ds1) - 1.0) * (29.0 * a1 * ds1 * 9.8) / 6.02 + p0
    if h < 20.0:
        return (
            (math.exp(-(h - 10.0) / ds2) - 1.0)
            * (29.0 * a2 * ds2 * 9.8)
            / 6.02
            + p1
        )
    return (
        (math.exp(-(h - 20.0) / ds3) - 1.0)
        * (29.0 * a3 * ds3 * 9.8)
        / 6.02
        + p2
    )


def neutral_molecular_concentration_cm3(altitude_km: float) -> float:
    h = altitude_km
    ds1, ds2, ds3 = 9.2080, 6.5059, 6.3779
    if h < 10.0:
        return 2.547e19 * math.exp(-h / ds1)
    if h < 20.0:
        return 8.597e18 * math.exp((10.0 - h) / ds2)
    return 1.849e18 * math.exp((20.0 - h) / ds3)


def build_atmosphere(config: AtmosphereConfig) -> AtmosphereState:
    temperature = temperature_profile(config.altitude_km)
    pressure = pressure_profile_hpa(config.altitude_km)
    molecular_concentration = neutral_molecular_concentration_cm3(config.altitude_km)

    t0 = 288.0
    p0 = 1013.2
    mobility_scale = (
        temperature**2.5 * ((t0 + 120.0) * p0)
        / (t0**2.5 * ((temperature + 120.0) * pressure))
    )
    mu_pos = config.positive_mobility_ref_m2_v_s * mobility_scale
    mu_neg = config.negative_mobility_ref_m2_v_s * mobility_scale
    d_pos = mu_pos * BOLTZMANN * temperature / ELEMENTARY_CHARGE
    d_neg = mu_neg * BOLTZMANN * temperature / ELEMENTARY_CHARGE

    return AtmosphereState(
        altitude_km=config.altitude_km,
        temperature_k=temperature,
        pressure_hpa=pressure,
        neutral_molecular_concentration_cm3=molecular_concentration,
        mobility_scale=mobility_scale,
        positive_mobility_m2_v_s=mu_pos,
        negative_mobility_m2_v_s=mu_neg,
        positive_diffusivity_m2_s=d_pos,
        negative_diffusivity_m2_s=d_neg,
    )
