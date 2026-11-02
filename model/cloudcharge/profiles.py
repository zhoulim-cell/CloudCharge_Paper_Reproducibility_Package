from __future__ import annotations

import numpy as np

from .config import DomainConfig, DropletConfig


def original_droplet_spectrum() -> tuple[float, float]:
    """Return number concentration and mean radius from the uploaded 0.5–30 μm spectrum."""
    radii = np.arange(0.5, 30.0 + 0.25, 0.5, dtype=float) * 1.0e-6
    da = 0.5e-6
    density = 2.7e24 * radii**2 * np.exp(-3.0e5 * radii)
    number = float(np.sum(da * density))
    radius_moment = float(np.sum(radii * da * density))
    return number, radius_moment / number


def peak_droplet_number(config: DropletConfig) -> float:
    if config.spectrum_mode == "original_gamma":
        number, _ = original_droplet_spectrum()
        return number
    if config.spectrum_mode == "fixed_peak":
        return config.fixed_peak_number_m3
    raise ValueError(f"Unknown droplet spectrum mode: {config.spectrum_mode}")


def droplet_radius_at_time(time_s: float, config: DropletConfig) -> float:
    if config.growth_time_s <= 0:
        return config.target_radius_m
    fraction = min(1.0, max(0.0, time_s / config.growth_time_s))
    return config.initial_radius_m + fraction * (
        config.target_radius_m - config.initial_radius_m
    )


def cloud_shape(z_m: np.ndarray, config: DomainConfig) -> np.ndarray:
    start = config.cloud_start_m
    ramp = config.edge_ramp_m
    core = config.core_width_m
    plateau_start = start + ramp
    plateau_end = plateau_start + core
    end = plateau_end + ramp

    shape = np.zeros_like(z_m, dtype=float)
    rising = (z_m >= start) & (z_m < plateau_start)
    plateau = (z_m >= plateau_start) & (z_m <= plateau_end)
    falling = (z_m > plateau_end) & (z_m <= end)

    shape[rising] = np.sin(0.5 * np.pi * (z_m[rising] - start) / ramp) ** 2
    shape[plateau] = 1.0
    shape[falling] = np.cos(0.5 * np.pi * (z_m[falling] - plateau_end) / ramp) ** 2
    return shape
