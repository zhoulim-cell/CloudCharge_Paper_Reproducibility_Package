from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cloudcharge.frontier_assimilation import ensemble_kalman_inversion
from cloudcharge.frontier_config import FrontierConfig
from cloudcharge.frontier_observations import make_synthetic_observations
from cloudcharge.frontier_solver import run_frontier_model


def reduced_config(path: Path, duration: float) -> FrontierConfig:
    cfg = FrontierConfig.load_json(path)
    cfg.base.domain.grid_points = 25
    cfg.base.spectrum.radius_bins = 4
    cfg.base.spectrum.radius_min_m = 4.0e-6
    cfg.base.spectrum.radius_max_m = 16.0e-6
    cfg.base.spectrum.charge_min_e = -35
    cfg.base.spectrum.charge_max_e = 35
    cfg.aerosol_spectrum.radius_bins = 3
    cfg.aerosol_spectrum.charge_min_e = -2
    cfg.aerosol_spectrum.charge_max_e = 2
    cfg.transport.include_droplet_sedimentation = False
    cfg.forcing.current_mode = "constant"
    cfg.forcing.ionization_mode = "constant"
    cfg.forcing.current_density_initial_a_m2 = 2.5e-12
    cfg.forcing.current_density_final_a_m2 = 2.5e-12
    cfg.forcing.step_time_s = duration + 1.0
    cfg.numerics.duration_s = duration
    cfg.numerics.max_dt_s = 0.5
    cfg.numerics.snapshot_times_s = [0.0, duration]
    cfg.numerics.diagnostics_interval_s = duration
    cfg.validate()
    return cfg


def main() -> None:
    p = argparse.ArgumentParser(description="Synthetic observation / ensemble inversion twin experiment")
    p.add_argument("--config", type=Path, default=Path("presets/frontier_baseline.json"))
    p.add_argument("--out", type=Path, default=Path("frontier_validation/twin_experiment"))
    p.add_argument("--duration", type=float, default=80.0)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    base = reduced_config(args.config, args.duration)
    truth = deepcopy(base)
    truth.forcing.current_density_initial_a_m2 *= 1.30
    truth.forcing.current_density_final_a_m2 *= 1.30
    truth.base.aerosol.total_number_m3 *= 0.75
    truth.transport.eddy_diffusivity_m2_s *= 1.20
    truth.transport.ion_eddy_diffusivity_m2_s *= 1.20
    truth_result = run_frontier_model(truth)
    obs = make_synthetic_observations(
        truth_result,
        args.duration,
        [5.0, 9.0, 14.0, 25.0, 29.0, 33.0],
        ["electric_field_v_m", "droplet_charge_e", "conductivity_s_m"],
        relative_noise=0.04,
        absolute_noise={"electric_field_v_m": 1.0, "droplet_charge_e": 0.25},
        seed=17,
    )
    obs.save_csv(args.out / "synthetic_observations.csv")

    base.assimilation.parameter_names = ["current_scale", "aerosol_scale", "eddy_diffusivity_scale"]
    base.assimilation.ensemble_size = 6
    base.assimilation.iterations = 2
    base.assimilation.prior_std_fraction = 0.30
    inversion = ensemble_kalman_inversion(
        base,
        obs,
        progress_callback=lambda f, m: print(f"{100*f:6.2f}% {m}"),
    )

    truth_values = np.array([1.30, 0.75, 1.20])
    posterior_mean = np.mean(inversion.posterior_ensemble, axis=0)
    posterior_std = np.std(inversion.posterior_ensemble, axis=0, ddof=1)
    with (args.out / "posterior_parameters.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["parameter", "truth", "posterior_mean", "posterior_std"])
        for name, tv, pm, ps in zip(inversion.parameter_names, truth_values, posterior_mean, posterior_std):
            w.writerow([name, tv, pm, ps])

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    x = np.arange(len(inversion.parameter_names))
    ax.errorbar(x, posterior_mean, yerr=posterior_std, fmt="o", capsize=5, label="posterior ensemble")
    ax.scatter(x, truth_values, marker="*", s=130, label="synthetic truth")
    ax.axhline(1.0, ls="--", lw=0.8, label="unscaled base")
    ax.set_xticks(x, inversion.parameter_names, rotation=15)
    ax.set_ylabel("Multiplicative scale")
    ax.set_title("Observation-constrained cloud-electric twin experiment")
    ax.grid(alpha=0.25); ax.legend()
    fig.savefig(args.out / "twin_parameter_recovery.png", dpi=200)
    plt.close(fig)

    summary = {
        "parameter_names": inversion.parameter_names,
        "truth": truth_values.tolist(),
        "posterior_mean": posterior_mean.tolist(),
        "posterior_std": posterior_std.tolist(),
        "note": "Derivative-free ensemble inversion twin experiment; not a formal Bayesian posterior.",
    }
    (args.out / "twin_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
