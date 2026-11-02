from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .solver import SimulationResult


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def export_npz(result: SimulationResult, path: str | Path) -> None:
    path = Path(path)
    payload: dict[str, np.ndarray] = {
        "z_m": result.z_m,
        "snapshot_times_s": result.snapshot_times_s,
    }
    for time_s, snapshot in result.snapshots.items():
        prefix = f"t_{time_s:g}_"
        for key, value in snapshot.items():
            payload[prefix + key] = np.asarray(value)
    for key, value in result.diagnostics.items():
        if isinstance(value, (np.ndarray, float, int)):
            payload["diag_" + key] = np.asarray(value)
    np.savez_compressed(path, **payload)


def export_snapshot_csv(result: SimulationResult, time_s: float, path: str | Path) -> None:
    time_s = result.nearest_snapshot_time(time_s)
    snapshot = result.snapshots[time_s]
    columns = ["z_m"]
    arrays = [result.z_m]
    for key, value in snapshot.items():
        arr = np.asarray(value)
        if arr.ndim == 1 and arr.size == result.z_m.size:
            columns.append(key)
            arrays.append(arr)
    table = np.column_stack(arrays)
    np.savetxt(path, table, delimiter=",", header=",".join(columns), comments="")


def export_metadata_json(result: SimulationResult, path: str | Path) -> None:
    content = {
        "metadata": result.metadata,
        "snapshot_times_s": result.snapshot_times_s,
        "diagnostics_summary": {
            key: value
            for key, value in result.diagnostics.items()
            if np.asarray(value).ndim == 0
        },
    }
    Path(path).write_text(
        json.dumps(content, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )


def export_spectrum_slice_csv(
    result: SimulationResult,
    time_s: float,
    z_m: float,
    path: str | Path,
) -> None:
    """Export one N(r,k) slice as a long CSV table."""
    time_s = result.nearest_snapshot_time(time_s)
    snapshot = result.snapshots[time_s]
    if "droplet_spectrum_m3" not in snapshot:
        raise ValueError("The selected result does not contain a full droplet spectrum.")
    z_index = int(np.argmin(np.abs(result.z_m - float(z_m))))
    spectrum = np.asarray(snapshot["droplet_spectrum_m3"], dtype=float)[z_index]
    radii = np.asarray(snapshot["radius_centers_m"], dtype=float)
    charges = np.asarray(snapshot["charge_states_e"], dtype=float)
    rr, qq = np.meshgrid(radii, charges, indexing="ij")
    table = np.column_stack(
        [
            np.full(rr.size, time_s),
            np.full(rr.size, result.z_m[z_index]),
            rr.ravel(),
            qq.ravel(),
            spectrum.ravel(),
        ]
    )
    np.savetxt(
        Path(path),
        table,
        delimiter=",",
        header="time_s,z_m,radius_m,charge_e,number_concentration_m3_per_bin",
        comments="",
    )
