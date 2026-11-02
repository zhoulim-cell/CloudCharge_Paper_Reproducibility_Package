from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cloudcharge.frontier_config import FrontierConfig
from cloudcharge.frontier_observations import make_synthetic_observations
from cloudcharge.frontier_solver import run_frontier_model
from cloudcharge.io import export_metadata_json, export_npz, export_snapshot_csv


def save_profile_figure(result, path: Path) -> None:
    t = float(result.snapshot_times_s[-1])
    s = result.snapshots[t]
    z = result.z_m
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    axes = axes.ravel()
    axes[0].plot(z, np.asarray(s["droplet_number_m3"]) / 1e6, label="droplets")
    axes[0].plot(z, np.asarray(s["aerosol_number_m3"]) / 1e6, label="aerosol")
    axes[0].set_ylabel("Number (cm$^{-3}$)"); axes[0].legend(); axes[0].set_title("Particle populations")
    axes[1].plot(z, s["electric_field_v_m"])
    axes[1].set_ylabel("E downward (V m$^{-1}$)"); axes[1].set_title("Electric field")
    axes[2].plot(z, np.asarray(s["charge_density_c_m3"]) * 1e12)
    axes[2].axhline(0, lw=.7); axes[2].set_ylabel("pC m$^{-3}$"); axes[2].set_title("Space charge")
    axes[3].plot(z, s["droplet_charge_e"], label="droplet")
    axes[3].plot(z, s["aerosol_charge_e"], label="aerosol")
    axes[3].axhline(0, lw=.7); axes[3].legend(); axes[3].set_ylabel("Mean charge (e)")
    axes[4].semilogy(z, np.maximum(s["n_positive_m3"], 1), label="n+")
    axes[4].semilogy(z, np.maximum(s["n_negative_m3"], 1), label="n-")
    axes[4].legend(); axes[4].set_ylabel("m$^{-3}$"); axes[4].set_title("Small ions")
    axes[5].plot(z, np.asarray(s["non_displacement_current_a_m2"]) / 1e-12, label="non-displacement")
    axes[5].plot(z, np.asarray(s["displacement_current_a_m2"]) / 1e-12, label="displacement")
    axes[5].plot(z, np.asarray(s["total_current_a_m2"]) / 1e-12, label="total")
    axes[5].axhline(float(s["target_current_a_m2"]) / 1e-12, ls="--", lw=.8, label="target")
    axes[5].legend(fontsize=8); axes[5].set_ylabel("pA m$^{-2}$"); axes[5].set_title("Current closure")
    for ax in axes:
        ax.set_xlabel("Distance downward (m)"); ax.grid(alpha=.25)
    fig.suptitle(f"Frontier cloud charging model, t={t:g} s")
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_time_figure(result, path: Path) -> None:
    d = result.diagnostics
    t = np.asarray(d["time_s"])
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes[0, 0].plot(t, d["top_droplet_charge_e"], label="cloud top")
    axes[0, 0].plot(t, d["bottom_droplet_charge_e"], label="cloud base")
    axes[0, 0].axhline(0, lw=.7); axes[0, 0].legend(); axes[0, 0].set_title("Droplet charge")
    axes[0, 1].plot(t, d["top_aerosol_charge_e"], label="cloud top")
    axes[0, 1].plot(t, d["bottom_aerosol_charge_e"], label="cloud base")
    axes[0, 1].axhline(0, lw=.7); axes[0, 1].legend(); axes[0, 1].set_title("Aerosol charge")
    axes[1, 0].plot(t, np.asarray(d["scheduled_current_density_a_m2"]) / 1e-12, label="Jz")
    axes[1, 0].plot(t, np.asarray(d["mean_displacement_current_fraction"]), label="|Jdisp|/Jz")
    axes[1, 0].legend(); axes[1, 0].set_title("Forcing and dielectric response")
    axes[1, 1].semilogy(t, np.maximum(d["max_relative_total_current_residual"], 1e-12), label="current")
    axes[1, 1].semilogy(t, np.maximum(d["max_relative_gauss_residual"], 1e-12), label="Gauss relative")
    axes[1, 1].legend(); axes[1, 1].set_title("Constraint diagnostics")
    for ax in axes.ravel():
        ax.set_xlabel("Time (s)"); ax.grid(alpha=.25)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_spectrum_figure(result, path: Path) -> None:
    t = float(result.snapshot_times_s[-1]); s = result.snapshots[t]
    z = result.z_m
    cloud_start = result.metadata["config"]["base"]["domain"]["cloud_start_m"]
    edge = result.metadata["config"]["base"]["domain"]["edge_ramp_m"]
    core = result.metadata["config"]["base"]["domain"]["core_width_m"]
    top_idx = int(np.argmin(np.abs(z - (cloud_start + 0.45 * edge))))
    base_idx = int(np.argmin(np.abs(z - (cloud_start + edge + core + 0.55 * edge))))
    dr = np.asarray(s["droplet_radius_centers_m"]) * 1e6
    dq = np.asarray(s["droplet_charge_states_e"])
    ar = np.asarray(s["aerosol_radius_centers_m"]) * 1e9
    aq = np.asarray(s["aerosol_charge_states_e"])
    ds = np.asarray(s["droplet_spectrum_m3"])
    ass = np.asarray(s["aerosol_spectrum_m3"])
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, spec, idx, title in [
        (axes[0, 0], ds, top_idx, "Droplets: cloud top"),
        (axes[0, 1], ds, base_idx, "Droplets: cloud base"),
    ]:
        im = ax.pcolormesh(dr, dq, np.log10(np.maximum(spec[idx].T, 1e-20)), shading="auto")
        ax.set_xscale("log"); ax.set_xlabel("Radius (µm)"); ax.set_ylabel("Charge (e)"); ax.set_title(title)
        fig.colorbar(im, ax=ax, label="log10 N per bin")
    for ax, spec, idx, title in [
        (axes[1, 0], ass, top_idx, "Aerosol: cloud top"),
        (axes[1, 1], ass, base_idx, "Aerosol: cloud base"),
    ]:
        im = ax.pcolormesh(ar, aq, np.log10(np.maximum(spec[idx].T, 1e-20)), shading="auto")
        ax.set_xscale("log"); ax.set_xlabel("Radius (nm)"); ax.set_ylabel("Charge (e)"); ax.set_title(title)
        fig.colorbar(im, ax=ax, label="log10 N per bin")
    fig.suptitle(f"Size–charge spectra at t={t:g} s")
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Run the frontier aerosol–ion–droplet cloud charge model.")
    p.add_argument("--config", type=Path, default=Path("presets/frontier_baseline.json"))
    p.add_argument("--out", type=Path, default=Path("frontier_output"))
    p.add_argument("--duration", type=float)
    p.add_argument("--synthetic-observations", action="store_true")
    args = p.parse_args()
    cfg = FrontierConfig.load_json(args.config)
    if args.duration is not None:
        cfg.numerics.duration_s = args.duration
        cfg.numerics.snapshot_times_s = sorted(set([0.0, args.duration]))
    cfg.validate()
    args.out.mkdir(parents=True, exist_ok=True)
    result = run_frontier_model(cfg, progress_callback=lambda f, m: print(f"{100*f:6.2f}% {m}"))
    export_npz(result, args.out / "frontier_result.npz")
    export_metadata_json(result, args.out / "frontier_metadata.json")
    export_snapshot_csv(result, float(result.snapshot_times_s[-1]), args.out / "frontier_final_profiles.csv")
    save_profile_figure(result, args.out / "frontier_profiles.png")
    save_time_figure(result, args.out / "frontier_time_evolution.png")
    save_spectrum_figure(result, args.out / "frontier_size_charge_spectra.png")
    if args.synthetic_observations:
        obs = make_synthetic_observations(
            result,
            float(result.snapshot_times_s[-1]),
            [3.0, 7.0, 12.0, 22.0, 28.0, 32.0],
            ["electric_field_v_m", "droplet_number_m3", "droplet_charge_e", "conductivity_s_m"],
            relative_noise=0.05,
            absolute_noise={"electric_field_v_m": 1.0, "droplet_charge_e": 1.0},
        )
        obs.save_csv(args.out / "synthetic_observations.csv")
    summary = {
        k: result.diagnostics[k]
        for k in [
            "upper_peak_droplet_charge_e",
            "lower_peak_droplet_charge_e",
            "upper_integrated_space_charge_c_m2",
            "lower_integrated_space_charge_c_m2",
            "opposite_droplet_sign_test",
            "opposite_space_charge_sign_test",
            "final_max_gauss_residual_v_m2",
            "final_max_relative_gauss_residual",
            "final_max_boundary_gauss_residual_v_m2",
            "final_max_relative_total_current_residual",
        ]
    }
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
