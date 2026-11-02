from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cloudcharge.config import SimulationConfig
from cloudcharge.io import export_metadata_json, export_npz, export_snapshot_csv, export_spectrum_slice_csv
from cloudcharge.solver import CloudChargeModel


def _save_profile_figure(result, output: Path) -> None:
    final_time = float(result.snapshot_times_s[-1])
    snap = result.snapshots[final_time]
    z = result.z_m
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    axes = axes.ravel()
    axes[0].plot(z, np.asarray(snap["droplet_number_m3"]) / 1.0e6)
    axes[0].set_title("Droplet number (cm$^{-3}$)")
    axes[1].plot(z, snap["electric_field_v_m"])
    axes[1].set_title("Electric field (V m$^{-1}$)")
    axes[2].plot(z, np.asarray(snap["charge_density_c_m3"]) * 1.0e12)
    axes[2].axhline(0.0, linewidth=0.7)
    axes[2].set_title("Space charge (pC m$^{-3}$)")
    axes[3].semilogy(z, np.maximum(snap["n_positive_m3"], 1.0), label="n+")
    axes[3].semilogy(z, np.maximum(snap["n_negative_m3"], 1.0), label="n-")
    axes[3].legend()
    axes[3].set_title("Small ions")
    axes[4].semilogy(z, np.maximum(snap["s_positive_m3"], 1.0), label="S+")
    axes[4].semilogy(z, np.maximum(snap["s_negative_m3"], 1.0), label="S-")
    axes[4].legend()
    axes[4].set_title("Charged aerosols")
    axes[5].plot(z, snap["droplet_charge_e"])
    axes[5].axhline(0.0, linewidth=0.7)
    axes[5].set_title("Droplet charge (e/drop)")
    for ax in axes:
        ax.set_xlabel("Distance downward (m)")
        ax.grid(True, alpha=0.25)
    fig.suptitle(f"Cloud charge model, t={final_time:g} s")
    fig.savefig(output, dpi=200)
    plt.close(fig)


def _save_time_figure(result, output: Path) -> bool:
    d = result.diagnostics
    if "time_s" not in d or "top_droplet_charge_e" not in d:
        return False
    t = np.asarray(d["time_s"], dtype=float)
    mode = str(d.get("solver_mode", ""))
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    axes = axes.ravel()
    axes[0].plot(t, d["top_droplet_charge_e"], label="cloud top")
    axes[0].plot(t, d["bottom_droplet_charge_e"], label="cloud base")
    axes[0].axhline(0.0, linewidth=0.7); axes[0].legend(); axes[0].set_title("Mean droplet charge")
    if mode == "paper2012_spectrum":
        axes[1].plot(t, d["top_positive_fraction"], label="top positive")
        axes[1].plot(t, d["bottom_negative_fraction"], label="base negative")
        axes[1].set_ylim(0, 1); axes[1].legend(); axes[1].set_title("Polarity fractions")
        axes[2].plot(t, np.asarray(d["top_mean_radius_m"]) * 1e6, label="top")
        axes[2].plot(t, np.asarray(d["bottom_mean_radius_m"]) * 1e6, label="base")
        axes[2].legend(); axes[2].set_title("Mean radius (um)")
    else:
        axes[1].plot(t, d["top_droplet_charging_rate_e_s"], label="cloud top")
        axes[1].plot(t, d["bottom_droplet_charging_rate_e_s"], label="cloud base")
        axes[1].axhline(0.0, linewidth=0.7); axes[1].legend(); axes[1].set_title("Droplet charging rate")
        axes[2].plot(t, d["top_positive_aerosol_m3"], label="S+")
        axes[2].plot(t, d["top_negative_aerosol_m3"], label="S-")
        axes[2].legend(); axes[2].set_title("Top-boundary charged aerosol")
    axes[3].plot(t, np.asarray(d["scheduled_current_density_a_m2"]) / 1e-12, label="Jz")
    axes[3].set_title("Scheduled current density")
    for ax in axes:
        ax.set_xlabel("Time (s)"); ax.grid(True, alpha=0.25)
    fig.savefig(output, dpi=200); plt.close(fig); return True


def _save_spectrum_figure(result, output: Path) -> bool:
    final_time = float(result.snapshot_times_s[-1]); snap = result.snapshots[final_time]
    if "droplet_spectrum_m3" not in snap:
        return False
    idx = int(float(result.diagnostics.get("top_spectrum_index", 0)))
    spectrum = np.asarray(snap["droplet_spectrum_m3"], float)[idx]
    radii = np.asarray(snap["radius_centers_m"], float) * 1e6
    charges = np.asarray(snap["charge_states_e"], float)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    image = axes[0].pcolormesh(radii, charges, np.log10(np.maximum(spectrum.T, 1e-20)), shading="auto")
    axes[0].set_xscale("log"); axes[0].set_xlabel("Radius (um)"); axes[0].set_ylabel("Charge (e)"); axes[0].set_title("log10 N(r,k)"); fig.colorbar(image, ax=axes[0])
    axes[1].plot(charges, spectrum.sum(axis=0)); axes[1].set_xlabel("Charge (e)"); axes[1].set_title("Charge marginal")
    axes[2].plot(radii, spectrum.sum(axis=1)); axes[2].set_xscale("log"); axes[2].set_xlabel("Radius (um)"); axes[2].set_title("Size marginal")
    fig.suptitle(f"t={final_time:g}s, z={result.z_m[idx]:.2f}m")
    fig.savefig(output, dpi=200); plt.close(fig); return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Zhou–Tinsley 1-D cloud charge model without the GUI.")
    parser.add_argument("--config", type=Path, help="JSON configuration file")
    parser.add_argument("--duration", type=float, help="Override duration in seconds")
    parser.add_argument("--out", type=Path, default=Path("output"), help="Output directory")
    args = parser.parse_args()

    cfg = SimulationConfig.load_json(args.config) if args.config else SimulationConfig()
    if args.duration is not None:
        if args.duration <= 0:
            parser.error("--duration must be positive")
        if cfg.physics.solution_mode in {"paper2012_time", "paper2012_direct", "paper2012_spectrum"}:
            if args.duration <= cfg.time_dependent.step_time_s:
                parser.error("For 2012 modes, --duration must exceed the configured step time")
            cfg.time_dependent.total_duration_s = args.duration
        cfg.numerics.duration_s = args.duration
        keep = [x for x in cfg.numerics.snapshot_times_s if 0.0 <= x <= args.duration]
        cfg.numerics.snapshot_times_s = sorted(set([0.0, *keep, args.duration]))
    cfg.validate()

    args.out.mkdir(parents=True, exist_ok=True)
    model = CloudChargeModel(cfg)
    result = model.run(progress_callback=lambda p, m: print(f"{100*p:6.2f}% {m}"))

    export_npz(result, args.out / "cloud_charge_result.npz")
    export_metadata_json(result, args.out / "cloud_charge_metadata.json")
    final_time = float(result.snapshot_times_s[-1])
    export_snapshot_csv(result, final_time, args.out / "cloud_charge_final.csv")
    _save_profile_figure(result, args.out / "cloud_charge_profiles.png")
    _save_time_figure(result, args.out / "cloud_charge_time_evolution.png")
    if _save_spectrum_figure(result, args.out / "cloud_charge_size_charge_spectrum.png"):
        top_index = int(float(result.diagnostics.get("top_spectrum_index", 0)))
        export_spectrum_slice_csv(result, final_time, float(result.z_m[top_index]), args.out / "cloud_charge_top_spectrum.csv")

    d = result.diagnostics
    summary = {
        "solver_mode": d.get("solver_mode"),
        "final_time_s": final_time,
        "cloud_top_peak_charge_e": float(d.get("upper_peak_droplet_charge_e", np.nan)),
        "cloud_base_peak_charge_e": float(d.get("lower_peak_droplet_charge_e", np.nan)),
        "opposite_droplet_sign_test": bool(d.get("opposite_droplet_sign_test", False)),
        "opposite_space_charge_sign_test": bool(d.get("opposite_space_charge_sign_test", False)),
        "response_tau_used_s": float(d.get("response_tau_used_s", np.nan)),
        "post_step_tau_top_s": float(d.get("post_step_tau_top_s", np.nan)),
        "post_step_tau_bottom_s": float(d.get("post_step_tau_bottom_s", np.nan)),
    }
    (args.out / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Outputs written to: {args.out.resolve()}")


if __name__ == "__main__":
    main()
