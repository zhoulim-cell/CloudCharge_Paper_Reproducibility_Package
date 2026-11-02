from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .frontier_config import FrontierConfig
from .frontier_observations import ObservationSet, observation_vector
from .frontier_solver import run_frontier_model


@dataclass
class InversionResult:
    parameter_names: list[str]
    prior_ensemble: np.ndarray
    posterior_ensemble: np.ndarray
    history_mean: np.ndarray
    history_std: np.ndarray
    predicted_observations: np.ndarray
    observation_values: np.ndarray
    observation_uncertainties: np.ndarray


def _apply_parameters(base: FrontierConfig, names: list[str], values: np.ndarray) -> FrontierConfig:
    cfg = deepcopy(base)
    for name, value in zip(names, values):
        value = float(max(value, 0.05))
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
        elif name == "drift_scale":
            # Boolean drift physics cannot be continuously scaled in the base
            # solver; use this parameter only in custom extensions.
            pass
        else:
            raise ValueError(f"Unknown inversion parameter {name}")
    cfg.validate()
    return cfg


def ensemble_kalman_inversion(
    base_config: FrontierConfig,
    observations: ObservationSet,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> InversionResult:
    """Derivative-free ensemble Kalman inversion for cloud-electric parameters.

    Parameters are represented as positive multiplicative factors around one.
    The routine is intended for twin experiments and field-campaign design;
    posterior uncertainty should not be interpreted as a formal Bayesian
    posterior without additional sampling and structural-error treatment.
    """
    acfg = base_config.assimilation
    names = list(acfg.parameter_names)
    ne = int(acfg.ensemble_size)
    rng = np.random.default_rng(acfg.random_seed)
    prior = np.exp(rng.normal(0.0, acfg.prior_std_fraction, size=(ne, len(names))))
    ensemble = prior.copy()
    history_mean = [np.mean(ensemble, axis=0)]
    history_std = [np.std(ensemble, axis=0, ddof=1)]
    R = observations.covariance()
    last_y = np.empty((ne, observations.value.size), dtype=float)
    total_runs = max(1, acfg.iterations * ne)
    run_index = 0
    for iteration in range(acfg.iterations):
        for member in range(ne):
            cfg = _apply_parameters(base_config, names, ensemble[member])
            # Only the requested observation time is required for inversion.
            cfg.numerics.duration_s = observations.time_s
            cfg.numerics.snapshot_times_s = [0.0, observations.time_s]
            cfg.numerics.diagnostics_interval_s = max(observations.time_s, 1.0)
            result = run_frontier_model(cfg)
            last_y[member] = observation_vector(result, observations)
            run_index += 1
            if progress_callback:
                progress_callback(
                    run_index / total_runs,
                    f"EnKI iteration {iteration+1}/{acfg.iterations}, member {member+1}/{ne}",
                )
        xmean = np.mean(ensemble, axis=0)
        ymean = np.mean(last_y, axis=0)
        X = ensemble - xmean
        Y = last_y - ymean
        Cxy = X.T @ Y / max(ne - 1, 1)
        Cyy = Y.T @ Y / max(ne - 1, 1) + R
        gain = Cxy @ np.linalg.pinv(Cyy)
        for member in range(ne):
            perturbed = observations.value + rng.multivariate_normal(np.zeros(observations.value.size), R)
            ensemble[member] = ensemble[member] + gain @ (perturbed - last_y[member])
        ensemble = np.maximum(ensemble, 0.05)
        mean = np.mean(ensemble, axis=0)
        ensemble = mean + acfg.inflation * (ensemble - mean)
        history_mean.append(np.mean(ensemble, axis=0))
        history_std.append(np.std(ensemble, axis=0, ddof=1))
    return InversionResult(
        parameter_names=names,
        prior_ensemble=prior,
        posterior_ensemble=ensemble,
        history_mean=np.asarray(history_mean),
        history_std=np.asarray(history_std),
        predicted_observations=last_y,
        observation_values=observations.value.copy(),
        observation_uncertainties=observations.uncertainty.copy(),
    )
