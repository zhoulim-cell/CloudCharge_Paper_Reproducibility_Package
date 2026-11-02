from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy.stats import qmc, spearmanr

from .frontier_config import FrontierConfig
from .frontier_solver import run_frontier_model


@dataclass
class UQResult:
    parameter_names: list[str]
    samples: np.ndarray
    outputs: dict[str, np.ndarray]
    spearman_correlation: dict[str, np.ndarray]


def _set_scale(cfg: FrontierConfig, name: str, value: float) -> None:
    if name == "current_scale":
        cfg.forcing.current_density_initial_a_m2 *= value
        cfg.forcing.current_density_final_a_m2 *= value
    elif name == "ionization_scale":
        cfg.forcing.ionization_initial_m3_s *= value
        cfg.forcing.ionization_final_m3_s *= value
    elif name == "aerosol_scale":
        cfg.base.aerosol.total_number_m3 *= value
    elif name == "droplet_scale":
        cfg.base.droplet.fixed_peak_number_m3 *= value
    elif name == "eddy_diffusivity_scale":
        cfg.transport.eddy_diffusivity_m2_s *= value
        cfg.transport.ion_eddy_diffusivity_m2_s *= value
    elif name == "edge_thickness_scale":
        cfg.base.domain.edge_ramp_m *= value
        cloud_end = cfg.base.domain.cloud_start_m + 2 * cfg.base.domain.edge_ramp_m + cfg.base.domain.core_width_m
        if cloud_end > cfg.base.domain.length_m:
            cfg.base.domain.length_m = cloud_end + 2.0
    else:
        raise ValueError(name)


def latin_hypercube_uq(
    base: FrontierConfig,
    ranges: dict[str, tuple[float, float]],
    sample_count: int = 24,
    seed: int = 42,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> UQResult:
    names = list(ranges)
    sampler = qmc.LatinHypercube(d=len(names), seed=seed)
    unit = sampler.random(sample_count)
    lower = np.array([ranges[n][0] for n in names], dtype=float)
    upper = np.array([ranges[n][1] for n in names], dtype=float)
    samples = qmc.scale(unit, lower, upper)
    output_names = [
        "upper_peak_droplet_charge_e",
        "lower_peak_droplet_charge_e",
        "upper_integrated_space_charge_c_m2",
        "lower_integrated_space_charge_c_m2",
        "final_max_relative_gauss_residual",
        "final_max_relative_total_current_residual",
    ]
    outputs = {name: np.empty(sample_count, dtype=float) for name in output_names}
    for i, row in enumerate(samples):
        cfg = deepcopy(base)
        for name, value in zip(names, row):
            _set_scale(cfg, name, float(value))
        cfg.validate()
        result = run_frontier_model(cfg)
        for output in output_names:
            outputs[output][i] = float(result.diagnostics[output])
        if progress_callback:
            progress_callback((i + 1) / sample_count, f"UQ sample {i+1}/{sample_count}")
    correlations: dict[str, np.ndarray] = {}
    for output, values in outputs.items():
        corr = np.empty(len(names), dtype=float)
        for j in range(len(names)):
            corr[j] = float(spearmanr(samples[:, j], values).statistic)
        correlations[output] = corr
    return UQResult(names, samples, outputs, correlations)
