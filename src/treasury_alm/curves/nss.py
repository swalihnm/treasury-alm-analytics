"""Nelson-Siegel-Svensson (NSS) zero-curve model fitted to par (constant-maturity) yields.

Why NSS: four interpretable factors (level b0, slope b1, curvature b2, second hump b3) with two
decay parameters, giving a smooth curve that can be evaluated at any maturity, which the IRRBB
engine needs for the 19 Basel bucket midpoints.

Fit design: the model describes the *zero* (spot) curve z(T), but FRED publishes *par* yields.
We therefore compute the par yield implied by z(T) (semi-annual coupons, bills <1Y treated as
zero-coupon money-market yields) and minimise the par-yield error. Fitting a zero-curve formula
directly to par yields would bias the long end.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

FloatArr = NDArray[np.float64]

# Months-to-years maturities of the FRED constant-maturity series
TENORS: dict[str, float] = {
    "gs1m": 1 / 12, "tb3ms": 0.25, "gs1": 1.0, "gs2": 2.0, "gs3": 3.0,
    "gs5": 5.0, "gs7": 7.0, "gs10": 10.0, "gs20": 20.0, "gs30": 30.0,
}


def _loadings(t: FloatArr, lam1: float, lam2: float) -> FloatArr:
    t = np.maximum(t, 1e-6)
    x1, x2 = t / lam1, t / lam2
    f1 = (1 - np.exp(-x1)) / x1
    f2 = f1 - np.exp(-x1)
    f3 = (1 - np.exp(-x2)) / x2 - np.exp(-x2)
    return np.column_stack([np.ones_like(t), f1, f2, f3])


@dataclass(frozen=True)
class NSSParams:
    b0: float
    b1: float
    b2: float
    b3: float
    tau1: float  # decay scale in years (lambda = 1/tau in some texts; here tau is a time scale)
    tau2: float

    def vector(self) -> FloatArr:
        return np.array([self.b0, self.b1, self.b2, self.b3, self.tau1, self.tau2])

    def zero(self, t: FloatArr | float) -> FloatArr:
        """Continuously-compounded zero rate in percent at maturity t (years)."""
        ta = np.atleast_1d(np.asarray(t, dtype=float))
        return _loadings(ta, self.tau1, self.tau2) @ np.array([self.b0, self.b1, self.b2, self.b3])

    def discount(self, t: FloatArr | float) -> FloatArr:
        ta = np.atleast_1d(np.asarray(t, dtype=float))
        return np.exp(-self.zero(ta) / 100.0 * ta)

    def par_yield(self, T: float, freq: int = 2) -> float:
        """Par yield (percent, bond-equivalent) implied by the zero curve."""
        if T <= 1.0:  # money-market instruments: simple-compounded zero yield
            d = float(self.discount(T)[0])
            return float((1 / d - 1) / T * 100.0)
        n = int(round(T * freq))
        times = np.arange(1, n + 1) / freq
        d = self.discount(times)
        return float(freq * (1 - d[-1]) / d.sum() * 100.0)


def par_yields(p: NSSParams, maturities: FloatArr) -> FloatArr:
    return np.array([p.par_yield(float(m)) for m in maturities])


BOUNDS_LO = np.array([0.0, -20.0, -30.0, -30.0, 0.2, 1.0])
BOUNDS_HI = np.array([25.0, 20.0, 30.0, 30.0, 5.0, 15.0])


def _init_linear(mats: FloatArr, y: FloatArr) -> FloatArr:
    """Grid over (tau1, tau2); OLS for betas treating yields as spot rates. Returns best start."""
    best, best_sse = None, np.inf
    for t1 in (0.5, 1.0, 2.0, 3.5):
        for t2 in (4.0, 8.0, 12.0):
            X = _loadings(mats, t1, t2)
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            sse = float(((X @ beta - y) ** 2).sum())
            if sse < best_sse:
                best, best_sse = np.concatenate([beta, [t1, t2]]), sse
    assert best is not None
    return np.clip(best, BOUNDS_LO + 1e-6, BOUNDS_HI - 1e-6)


def fit_nss(mats: FloatArr, y: FloatArr, start: FloatArr | None = None) -> tuple[NSSParams, float]:
    """Bounded least-squares NSS fit to par yields (percent). Returns (params, RMSE in bp).

    Multi-start: linear-OLS grid start plus an optional warm start (previous month) for stability.
    Needs >= 6 maturities (6 parameters); constraint b0 + b1 > 0 (short rate positive-ish) enforced
    softly by a penalty residual.
    """
    if len(mats) < 6:
        raise ValueError("need at least 6 maturities")

    def resid(v: FloatArr) -> FloatArr:
        p = NSSParams(*v)
        r = par_yields(p, mats) - y
        pen = max(0.0, -(v[0] + v[1]) + 1e-3) * 10.0  # short-rate non-negativity (soft)
        return np.append(r, pen)

    starts = [_init_linear(mats, y)]
    if start is not None:
        starts.append(np.clip(start, BOUNDS_LO + 1e-6, BOUNDS_HI - 1e-6))
    best_v, best_c = None, np.inf
    for s in starts:
        sol = least_squares(resid, s, bounds=(BOUNDS_LO, BOUNDS_HI), method="trf", max_nfev=200)
        if sol.cost < best_c:
            best_v, best_c = sol.x, sol.cost
    assert best_v is not None
    p = NSSParams(*best_v)
    rmse_bp = float(np.sqrt(np.mean((par_yields(p, mats) - y) ** 2)) * 100.0)
    return p, rmse_bp
