from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .constants import ELEMENTARY_CHARGE
from .solver import SimulationResult


SUPPORTED_VARIABLES = (
    "electric_field_v_m",
    "charge_density_c_m3",
    "n_positive_m3",
    "n_negative_m3",
    "droplet_number_m3",
    "droplet_mean_radius_m",
    "droplet_charge_e",
    "aerosol_number_m3",
    "aerosol_charge_e",
    "conductivity_s_m",
)


@dataclass
class ObservationSet:
    time_s: float
    height_m: np.ndarray
    variable: np.ndarray
    value: np.ndarray
    uncertainty: np.ndarray

    def vector(self) -> np.ndarray:
        return np.asarray(self.value, dtype=float)

    def covariance(self) -> np.ndarray:
        return np.diag(np.asarray(self.uncertainty, dtype=float) ** 2)

    def save_csv(self, path: str | Path) -> None:
        with Path(path).open("w", encoding="utf-8") as f:
            f.write("time_s,height_m,variable,value,uncertainty\n")
            for h, v, y, s in zip(self.height_m, self.variable, self.value, self.uncertainty):
                f.write(f"{self.time_s:.9g},{float(h):.9g},{v},{float(y):.12g},{float(s):.12g}\n")

    @classmethod
    def load_csv(cls, path: str | Path) -> "ObservationSet":
        data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
        times = np.asarray(data["time_s"], dtype=float)
        if np.ptp(times) > 1.0e-9:
            raise ValueError("One ObservationSet CSV must contain a single observation time.")
        return cls(
            time_s=float(times.flat[0]),
            height_m=np.atleast_1d(np.asarray(data["height_m"], dtype=float)),
            variable=np.atleast_1d(np.asarray(data["variable"], dtype=str)),
            value=np.atleast_1d(np.asarray(data["value"], dtype=float)),
            uncertainty=np.atleast_1d(np.asarray(data["uncertainty"], dtype=float)),
        )


def _snapshot(result: SimulationResult, time_s: float) -> dict[str, np.ndarray | float]:
    t = result.nearest_snapshot_time(time_s)
    return result.snapshots[t]


def _variable_profile(
    result: SimulationResult,
    snapshot: dict[str, np.ndarray | float],
    variable: str,
) -> np.ndarray:
    if variable not in SUPPORTED_VARIABLES:
        raise ValueError(f"Unsupported observation variable: {variable}")
    if variable == "conductivity_s_m":
        cfg = result.metadata["config"]
        # Mobility values are saved in atmosphere metadata by the frontier solver.
        atmosphere = result.metadata["atmosphere"]
        mu_p = float(atmosphere["positive_mobility_m2_v_s"])
        mu_n = float(atmosphere["negative_mobility_m2_v_s"])
        return ELEMENTARY_CHARGE * (
            mu_p * np.asarray(snapshot["n_positive_m3"], dtype=float)
            + mu_n * np.asarray(snapshot["n_negative_m3"], dtype=float)
        )
    return np.asarray(snapshot[variable], dtype=float)


def observation_vector(
    result: SimulationResult,
    observations: ObservationSet,
) -> np.ndarray:
    snap = _snapshot(result, observations.time_s)
    output = np.empty(observations.value.size, dtype=float)
    for i, (height, variable) in enumerate(zip(observations.height_m, observations.variable)):
        profile = _variable_profile(result, snap, str(variable))
        output[i] = np.interp(float(height), result.z_m, profile)
    return output


def make_synthetic_observations(
    result: SimulationResult,
    time_s: float,
    heights_m: list[float] | np.ndarray,
    variables: list[str],
    relative_noise: float = 0.05,
    absolute_noise: dict[str, float] | None = None,
    seed: int = 42,
) -> ObservationSet:
    heights = np.asarray(heights_m, dtype=float)
    rows_h: list[float] = []
    rows_v: list[str] = []
    rows_y: list[float] = []
    rows_s: list[float] = []
    snap = _snapshot(result, time_s)
    absn = absolute_noise or {}
    rng = np.random.default_rng(seed)
    for variable in variables:
        profile = _variable_profile(result, snap, variable)
        for h in heights:
            truth = float(np.interp(h, result.z_m, profile))
            sigma = max(abs(truth) * relative_noise, float(absn.get(variable, 0.0)), 1.0e-30)
            rows_h.append(float(h))
            rows_v.append(variable)
            rows_y.append(float(truth + rng.normal(0.0, sigma)))
            rows_s.append(sigma)
    return ObservationSet(
        time_s=float(time_s),
        height_m=np.asarray(rows_h),
        variable=np.asarray(rows_v, dtype=str),
        value=np.asarray(rows_y),
        uncertainty=np.asarray(rows_s),
    )
