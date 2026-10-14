"""Common interface for deposit forecasters.

All forecasters predict r_{t,h} (h-month cumulative log growth, %) for a set of origin months.
Leakage rule: `predict` may only read rows of `df` up to and including each origin; the test
suite checks this by truncating `df` after the origin and asserting identical output.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from ..features import HORIZONS


class BaseForecaster(ABC):
    name: str = "base"

    def __init__(self) -> None:
        self._calib: dict[int, np.ndarray] = {}

    @abstractmethod
    def fit(self, df: pd.DataFrame, cutoff: pd.Timestamp) -> BaseForecaster:
        """Fit using only information up to `cutoff` (targets must also resolve by `cutoff`)."""

    @abstractmethod
    def predict(self, df: pd.DataFrame, origins: pd.DatetimeIndex) -> pd.DataFrame:
        """Return DataFrame indexed by origin with one column per horizon in HORIZONS."""

    def set_calibration(self, abs_resid: dict[int, np.ndarray]) -> None:
        """Per-horizon absolute out-of-sample residuals (already resolved at forecast time)."""
        self._calib = abs_resid

    def predict_interval(self, df: pd.DataFrame, origins: pd.DatetimeIndex, level: float = 0.8) -> dict[int, pd.DataFrame]:
        """Split-conformal intervals: point +/- the ceil((n+1)*level)-th smallest calibration |residual|."""
        pred = self.predict(df, origins)
        out: dict[int, pd.DataFrame] = {}
        for h in HORIZONS:
            r = np.sort(self._calib.get(h, np.array([])))
            n = len(r)
            k = int(np.ceil((n + 1) * level))
            q = r[k - 1] if n > 0 and k <= n else np.inf
            out[h] = pd.DataFrame({"lo": pred[h] - q, "hi": pred[h] + q})
        return out


def conformal_quantile(abs_resid: np.ndarray, level: float) -> float:
    r = np.sort(np.asarray(abs_resid, dtype=float))
    n = len(r)
    k = int(np.ceil((n + 1) * level))
    return float(r[k - 1]) if n > 0 and k <= n else float("inf")
