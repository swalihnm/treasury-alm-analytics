"""Zero-coupon curve object used for discounting and the 19 Basel IRRBB time-bucket midpoints.

Bucket midpoints follow the Basel Framework SRP31 standardised-framework time-bucket table
(SRP31.? "Time buckets and midpoints": overnight 0.0028y, O/N-1M 0.0417y, 1M-3M 0.1667y,
3M-6M 0.375y, 6M-9M 0.625y, 9M-1Y 0.875y, 1-1.5Y 1.25y, 1.5-2Y 1.75y, 2-3Y 2.5y, then 3.5..9.5y
yearly, 10-15Y 12.5y, 15-20Y 17.5y, >20Y 25y). See config/irrbb_basel.yaml for the audited copy.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .nss import FloatArr, NSSParams

BASEL_MIDPOINTS: list[float] = [0.0028, 0.0417, 0.1667, 0.375, 0.625, 0.875, 1.25, 1.75, 2.5, 3.5, 4.5,
                                5.5, 6.5, 7.5, 8.5, 9.5, 12.5, 17.5, 25.0]


@dataclass
class ZeroCurve:
    """Continuously compounded zero rates (percent) on a maturity grid, linearly interpolated."""

    tenors: FloatArr
    rates: FloatArr  # percent

    @classmethod
    def from_nss(cls, p: NSSParams, grid: FloatArr | None = None) -> ZeroCurve:
        g = np.array(sorted(set(BASEL_MIDPOINTS) | set(np.arange(0.25, 30.01, 0.25))), dtype=float) if grid is None else grid
        return cls(g, p.zero(g))

    def zero(self, t: FloatArr | float) -> NDArray[np.float64]:
        ta = np.atleast_1d(np.asarray(t, dtype=float))
        return np.interp(ta, self.tenors, self.rates)

    def discount(self, t: FloatArr | float) -> NDArray[np.float64]:
        ta = np.atleast_1d(np.asarray(t, dtype=float))
        return np.exp(-self.zero(ta) / 100.0 * ta)

    def forward(self, t1: float, t2: float) -> float:
        d1, d2 = float(self.discount(t1)[0]), float(self.discount(t2)[0])
        return float(np.log(d1 / d2) / (t2 - t1) * 100.0)

    def shifted(self, shock_bp: FloatArr) -> ZeroCurve:
        """New curve with a tenor-specific shock (bp) added to the zero rates at self.tenors."""
        return ZeroCurve(self.tenors.copy(), self.rates + shock_bp / 100.0)

    def price_par_bond(self, T: float, coupon_pct: float, freq: int = 2) -> float:
        """PV per 100 face of a coupon bond discounted on this curve (== 100 when coupon is the par yield)."""
        n = int(round(T * freq))
        times = np.arange(1, n + 1) / freq
        cf = np.full(n, coupon_pct / freq)
        cf[-1] += 100.0
        return float((cf * self.discount(times)).sum())

    def par_yield(self, T: float, freq: int = 2) -> float:
        n = int(round(T * freq))
        times = np.arange(1, n + 1) / freq
        d = self.discount(times)
        return float(freq * (1 - d[-1]) / d.sum() * 100.0)

    def bucket_discount_factors(self) -> FloatArr:
        return self.discount(np.array(BASEL_MIDPOINTS))
