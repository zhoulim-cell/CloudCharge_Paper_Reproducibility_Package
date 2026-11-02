from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cloudcharge.frontier_config import FrontierConfig
from cloudcharge.frontier_uq import latin_hypercube_uq


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
    cfg.forcing.current_density_initial_a_m2 = 3.0e-12
    cfg.forcing.current_density_final_a_m2 = 3.0e-12
    cfg.forcing.step_time_s = duration + 1.0
    cfg.numerics.duration_s = duration
    cfg.numerics.max_dt_s = 0.5
    cfg.numerics.snapshot_times_s = [0.0, duration]
    cfg.numerics.diagnostics_interval_s = duration
    cfg.validate()
    return cfg


def main() -> None:
    p = argparse.ArgumentParser(description="Latin-hypercube uncertainty and sensitivity experiment")
    p.add_argument("--config", type=Path, default=Path("presets/frontier_baseline.json"))
    p.add_argument("--out", type=Path, default=Path("frontier_validation/uq"))
    p.add_argument("--duration", type=float, default=80.0)
    p.add_argument("--samples", type=int, default=10)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = reduced_config(args.config, args.duration)
    ranges = {
        "current_scale": (0.6, 1.4),
        "ionization_scale": (0.6, 1.4),
        "aerosol_scale": (0.5, 2.0),
        "eddy_diffusivity_scale": (0.4, 2.0),
        "edge_thickness_scale": (0.6, 1.4),
    }
    result = latin_hypercube_uq(
        cfg, ranges, sample_count=args.samples, seed=23,
        progress_callback=lambda f, m: print(f"{100*f:6.2f}% {m}"),
    )
    with (args.out / "uq_samples_outputs.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        output_names = list(result.outputs)
        w.writerow(result.parameter_names + output_names)
        for i in range(args.samples):
            w.writerow(result.samples[i].tolist() + [result.outputs[n][i] for n in output_names])
    with (args.out / "spearman_sensitivity.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["output"] + result.parameter_names)
        for output, corr in result.spearman_correlation.items():
            w.writerow([output] + corr.tolist())

    key = "upper_peak_droplet_charge_e"
    corr = result.spearman_correlation[key]
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.bar(result.parameter_names, corr)
    ax.axhline(0, lw=0.8)
    ax.set_ylim(-1, 1)
    ax.set_ylabel("Spearman rank correlation")
    ax.set_title("Sensitivity of upper-boundary droplet charge")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(args.out / "uq_sensitivity_upper_charge.png", dpi=200)
    plt.close(fig)
    summary = {"sample_count": args.samples, "ranges": ranges, "key_output": key, "correlation": dict(zip(result.parameter_names, corr.tolist()))}
    (args.out / "uq_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
