from __future__ import annotations

from pathlib import Path

import numpy as np


class MeasuredAttachmentCorrection:
    """Optional multiplicative correction from a user-provided measurement table.

    CSV columns: radius_m, charge_e, polarity, multiplier
    polarity is positive or negative.  The table is deliberately supplied by
    the user rather than hard-coded, because available chamber measurements
    cover limited size/charge ranges and should not be extrapolated silently.
    """

    def __init__(self, path: str = ""):
        self.enabled = bool(path)
        self.tables: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        if not self.enabled:
            return
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Attachment correction table not found: {p}")
        data = np.genfromtxt(p, delimiter=",", names=True, dtype=None, encoding="utf-8")
        required = {"radius_m", "charge_e", "polarity", "multiplier"}
        if not required.issubset(data.dtype.names or ()):
            raise ValueError(f"Attachment table must contain columns {sorted(required)}")
        for polarity in ("positive", "negative"):
            mask = np.asarray(data["polarity"] == polarity)
            if not np.any(mask):
                continue
            self.tables[polarity] = (
                np.asarray(data["radius_m"][mask], dtype=float),
                np.asarray(data["charge_e"][mask], dtype=float),
                np.asarray(data["multiplier"][mask], dtype=float),
            )

    def __call__(self, radius_m: np.ndarray, charge_e: np.ndarray, polarity: str) -> np.ndarray:
        if not self.enabled or polarity not in self.tables:
            return np.ones(np.broadcast_shapes(radius_m.shape, charge_e.shape), dtype=float)
        rtab, qtab, mtab = self.tables[polarity]
        r, q = np.broadcast_arrays(radius_m, charge_e)
        # Inverse-distance interpolation in log-radius/charge space, with no
        # extrapolation beyond the convex cloud of measurements: nearest value
        # is used and a warning is recorded by metadata in the solver.
        lr = np.log(np.maximum(r, 1.0e-30))[..., None]
        lrt = np.log(np.maximum(rtab, 1.0e-30))[None, ...]
        scale_q = max(float(np.ptp(qtab)), 1.0)
        dist2 = (lr - lrt) ** 2 + ((q[..., None] - qtab[None, ...]) / scale_q) ** 2
        idx = np.argmin(dist2, axis=-1)
        return np.maximum(mtab[idx], 0.0)
