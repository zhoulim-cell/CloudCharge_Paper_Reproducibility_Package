from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cloudcharge.config import SimulationConfig
from cloudcharge.io import export_metadata_json, export_npz, export_snapshot_csv
from cloudcharge.solver import CloudChargeModel

OUT = Path(__file__).resolve().parent / "paper_validation"
OUT.mkdir(exist_ok=True)


def cfg2007() -> SimulationConfig:
    c = SimulationConfig()
    c.physics.solution_mode = "paper2007_bvp"
    c.atmosphere.altitude_km = 4.0
    c.atmosphere.ion_pair_production_m3_s = 8.0e6
    c.atmosphere.current_density_a_m2 = 3.0e-12
    c.atmosphere.recombination_m3_s = 1.65e-12
    c.aerosol.total_number_m3 = 5.0e9
    c.droplet.initial_radius_m = 10.0e-6
    c.droplet.target_radius_m = 10.0e-6
    c.droplet.growth_time_s = 0.0
    c.domain.grid_points = 181
    c.numerics.duration_s = 600.0
    c.numerics.snapshot_times_s = [0.0, 600.0]
    return c


def cfg2012() -> SimulationConfig:
    c = SimulationConfig()
    c.physics.solution_mode = "paper2012_time"
    c.atmosphere.altitude_km = 4.0
    c.atmosphere.recombination_m3_s = 1.65e-12
    c.aerosol.total_number_m3 = 1.0e9
    c.droplet.initial_radius_m = 0.01e-6
    c.droplet.target_radius_m = 10.0e-6
    c.droplet.growth_time_s = 600.0
    c.domain.grid_points = 181
    c.time_dependent.initial_current_density_a_m2 = 1.5e-12
    c.time_dependent.second_current_density_a_m2 = 3.0e-12
    c.time_dependent.initial_ion_pair_production_m3_s = 8.0e6
    c.time_dependent.second_ion_pair_production_m3_s = 8.0e6
    c.time_dependent.step_time_s = 5000.0
    c.time_dependent.total_duration_s = 10000.0
    c.numerics.duration_s = 10000.0
    c.numerics.max_dt_s = 10.0
    c.numerics.diagnostics_interval_s = 10.0
    c.numerics.snapshot_times_s = [0.0, 600.0, 2000.0, 5000.0, 6000.0, 10000.0]
    return c


def run_case(c: SimulationConfig):
    c.validate()
    return CloudChargeModel(c).run()


r07 = run_case(cfg2007())
r12 = run_case(cfg2012())

export_npz(r07, OUT / "paper2007_result.npz")
export_metadata_json(r07, OUT / "paper2007_metadata.json")
export_snapshot_csv(r07, 600.0, OUT / "paper2007_profile.csv")
export_npz(r12, OUT / "paper2012_result.npz")
export_metadata_json(r12, OUT / "paper2012_metadata.json")
for t in r12.snapshot_times_s:
    export_snapshot_csv(r12, float(t), OUT / f"paper2012_profile_t{t:g}.csv")

# Single-panel figures for publication-style checking.
fig, ax = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
ax.plot(r07.z_m, r07.snapshots[600.0]["droplet_charge_e"], label="2007 equilibrium")
ax.plot(r12.z_m, r12.snapshots[5000.0]["droplet_charge_e"], label="2012 before Jz step")
ax.plot(r12.z_m, r12.snapshots[10000.0]["droplet_charge_e"], label="2012 after Jz step")
ax.axhline(0.0, linewidth=0.8)
ax.set_xlabel("Distance downward through cloud (m)")
ax.set_ylabel("Mean droplet charge (e/drop)")
ax.set_title("Cloud-top positive and cloud-base negative droplet charge")
ax.grid(True, alpha=0.25)
ax.legend()
fig.savefig(OUT / "paper2007_2012_charge_profiles.png", dpi=220)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
t = np.asarray(r12.diagnostics["time_s"])
ax.plot(t, r12.diagnostics["top_droplet_charge_e"], label="cloud top")
ax.plot(t, r12.diagnostics["bottom_droplet_charge_e"], label="cloud base")
ax.axvline(5000.0, linestyle="--", linewidth=0.9, label="Jz step")
ax.axhline(0.0, linewidth=0.8)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Mean droplet charge (e/drop)")
ax.set_title("2012 time-dependent droplet charging")
ax.grid(True, alpha=0.25)
ax.legend()
fig.savefig(OUT / "paper2012_time_evolution.png", dpi=220)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
ax.plot(t, r12.diagnostics["top_droplet_charging_rate_e_s"], label="cloud top")
ax.plot(t, r12.diagnostics["bottom_droplet_charging_rate_e_s"], label="cloud base")
ax.axvline(5000.0, linestyle="--", linewidth=0.9, label="Jz step")
ax.axhline(0.0, linewidth=0.8)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Charging rate dp/dt (e/s)")
ax.set_title("Charging-rate response and decay")
ax.grid(True, alpha=0.25)
ax.legend()
fig.savefig(OUT / "paper2012_charging_rate.png", dpi=220)
plt.close(fig)


def paper_case(name: str, *, alt=4.0, q=8.0e6, st=1.0e9, j1=1.5e-12, j2=3.0e-12, radius=10.0e-6, q2=None):
    c = cfg2012()
    c.atmosphere.altitude_km = alt
    c.aerosol.total_number_m3 = st
    c.droplet.target_radius_m = radius
    c.time_dependent.initial_ion_pair_production_m3_s = q
    c.time_dependent.second_ion_pair_production_m3_s = q if q2 is None else q2
    c.time_dependent.initial_current_density_a_m2 = j1
    c.time_dependent.second_current_density_a_m2 = j2
    if alt == 2.0:
        c.atmosphere.recombination_m3_s = 1.66e-12
    elif alt == 10.0:
        c.atmosphere.recombination_m3_s = 1.78e-12
    result = run_case(c)
    d = result.diagnostics
    return {
        "case": name,
        "model_tau_s": float(d["response_tau_used_s"]),
        "fitted_tau_top_s": float(d["post_step_tau_top_s"]),
        "model_top_10000_e": float(d["upper_peak_droplet_charge_e"]),
        "model_base_10000_e": float(d["lower_peak_droplet_charge_e"]),
        "paper_case": str(d["paper_reference_case"]),
        "paper_tau_s": float(d["paper_reference_tau_s"]),
        "paper_top_10000_e": float(d["paper_reference_top_e"]),
        "paper_base_10000_e": float(d["paper_reference_base_e"]),
    }


cases = [
    paper_case("4 km baseline"),
    paper_case("2 km clean", alt=2.0, q=4.0e6),
    paper_case("2 km polluted", alt=2.0, q=4.0e6, st=20.0e9),
    paper_case("10 km low q", alt=10.0, q=15.0e6),
    paper_case("10 km high q", alt=10.0, q=40.0e6, j1=2.0e-12, j2=4.0e-12),
    paper_case("5 um droplets", radius=5.0e-6),
    paper_case("15 um droplets", radius=15.0e-6),
    paper_case("q and Jz reduced 30%", st=5.0e9, j2=2.1e-12, q2=5.6e6),
]
with (OUT / "paper2012_case_validation.csv").open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=list(cases[0]))
    writer.writeheader()
    writer.writerows(cases)

summary = {
    "paper2007": {
        "top_e": float(r07.diagnostics["upper_peak_droplet_charge_e"]),
        "base_e": float(r07.diagnostics["lower_peak_droplet_charge_e"]),
        "top_surface_charge_c_m2": float(r07.diagnostics["upper_integrated_space_charge_c_m2"]),
        "base_surface_charge_c_m2": float(r07.diagnostics["lower_integrated_space_charge_c_m2"]),
    },
    "paper2012_baseline": {
        "top_10000_e": float(r12.diagnostics["upper_peak_droplet_charge_e"]),
        "base_10000_e": float(r12.diagnostics["lower_peak_droplet_charge_e"]),
        "tau_used_s": float(r12.diagnostics["response_tau_used_s"]),
        "tau_fitted_top_s": float(r12.diagnostics["post_step_tau_top_s"]),
        "tau_fitted_base_s": float(r12.diagnostics["post_step_tau_bottom_s"]),
    },
    "cases": cases,
}
(OUT / "validation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
