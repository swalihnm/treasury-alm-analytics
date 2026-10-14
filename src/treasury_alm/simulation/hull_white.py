"""One-factor Hull-White short-rate model:  dr = (theta(t) - a r) dt + sigma dW.

Why Hull-White: mean reversion keeps rates in a plausible range, and the time-dependent drift
theta(t) is chosen so the model reproduces TODAY'S curve exactly (here the Module 2 NSS zero curve),
which Vasicek cannot do. Writing r_t = x_t + alpha(t) with x an OU process started at 0 and
    alpha(t) = f(0,t) + sigma^2/(2a^2) (1 - e^{-a t})^2
gives the exact discretisation used below. Zero-coupon prices on any path:
    P(t,T) = A(t,T) exp(-B(t,T) r_t),  B = (1 - e^{-a(T-t)})/a,
    ln A = ln[P(0,T)/P(0,t)] + B f(0,t) - sigma^2/(4a) (1 - e^{-2at}) B^2.

(a, sigma) are estimated from history by the exact-discretisation AR(1) of the OU process
(`calibrate_ou`); the curve fit is exact by construction and checked by a Monte Carlo martingale test.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from numpy.typing import NDArray
from scipy import stats

Arr = NDArray[np.float64]


def calibrate_ou(series_pct: pd.Series, dt: float = 1 / 12) -> dict[str, Any]:
    """AR(1) fit r_{t+1} = c + phi r_t + e.  a = -ln(phi)/dt;  sigma^2 = Var(e) * 2a / (1 - phi^2)."""
    s = series_pct.dropna() / 100.0
    y, x = s.iloc[1:].values, sm.add_constant(s.iloc[:-1].values)
    res = sm.OLS(y, x).fit()
    c, phi = float(res.params[0]), float(res.params[1])
    a = -np.log(phi) / dt
    sig = float(np.sqrt(res.mse_resid * 2 * a / (1 - phi**2)))
    jb = stats.jarque_bera(res.resid)
    return {
        "a": float(a),
        "sigma": sig,
        "long_run_mean_pct": 100 * c / (1 - phi),
        "phi": phi,
        "phi_se": float(res.bse[1]),
        "half_life_years": float(np.log(2) / a),
        "n_obs": int(res.nobs),
        "start": str(s.index[0].date()),
        "end": str(s.index[-1].date()),
        "resid_jarque_bera_p": float(jb.pvalue),
        "resid_excess_kurtosis": float(stats.kurtosis(res.resid)),
        "sigma_bp_per_sqrt_year": 1e4 * sig,
    }


@dataclass
class HullWhite:
    a: float
    sigma: float
    zero_pct: Callable[[Arr], Arr]  # continuously compounded zero rate in percent at maturities t (years)

    def lnP0(self, t: Arr) -> Arr:
        t = np.asarray(t, dtype=float)
        return -self.zero_pct(t) / 100.0 * t

    def fwd(self, t: Arr, h: float = 1e-4) -> Arr:
        """Instantaneous forward rate f(0,t) = -d ln P(0,t)/dt (decimal), central difference."""
        t = np.asarray(t, dtype=float)
        lo = np.maximum(t - h, 0.0)
        return np.asarray(-(self.lnP0(t + h) - self.lnP0(lo)) / (t + h - lo), dtype=np.float64)

    def alpha(self, t: Arr) -> Arr:
        t = np.asarray(t, dtype=float)
        return self.fwd(t) + self.sigma**2 / (2 * self.a**2) * (1 - np.exp(-self.a * t)) ** 2

    def simulate(
        self, n_paths: int, n_months: int, seed: int, antithetic: bool = True, deterministic: bool = False
    ) -> Arr:
        """Short rate paths r[k] (decimal) at t_k = k/12, k = 0..n_months. deterministic=True gives the forward path alpha(t)."""
        dt = 1 / 12
        t = np.arange(n_months + 1) * dt
        al = self.alpha(np.asarray(t, dtype=np.float64))
        if deterministic:
            return np.tile(al, (1, 1))
        rng = np.random.default_rng(seed)
        half = n_paths // 2 if antithetic else n_paths
        z = rng.standard_normal((half, n_months))
        if antithetic:
            z = np.stack([z, -z], axis=1).reshape(-1, n_months)  # interleaved: any prefix stays antithetic-balanced
        decay = np.exp(-self.a * dt)
        sd = self.sigma * np.sqrt((1 - np.exp(-2 * self.a * dt)) / (2 * self.a))
        x = np.zeros((z.shape[0], n_months + 1))
        for k in range(n_months):
            x[:, k + 1] = x[:, k] * decay + sd * z[:, k]
        return x + al

    def zero_rates(self, r: Arr, tenor: float) -> Arr:
        """Continuously compounded zero rate (percent) for a fixed time-to-maturity at every (path, month)."""
        n_months = r.shape[1] - 1
        t = np.arange(n_months + 1) / 12.0
        B = (1 - np.exp(-self.a * tenor)) / self.a
        lnA = (
            self.lnP0(t + tenor)
            - self.lnP0(t)
            + B * self.fwd(t)
            - self.sigma**2 / (4 * self.a) * (1 - np.exp(-2 * self.a * t)) * B**2
        )
        return np.asarray(-(lnA - B * r) / tenor * 100.0, dtype=np.float64)

    def martingale_check(self, n_paths: int, maturity: float, seed: int) -> dict[str, float]:
        """E[exp(-int r dt)] from simulation vs the curve's P(0,T): the HW 'fits today's curve' test."""
        n = int(round(maturity * 12))
        r = self.simulate(n_paths, n, seed)
        integral = (0.5 * (r[:, :-1] + r[:, 1:]) / 12.0).sum(axis=1)
        mc = float(np.exp(-integral).mean())
        se = float(np.exp(-integral).std() / np.sqrt(len(integral)))
        mkt = float(np.exp(self.lnP0(np.array([maturity]))[0]))
        return {
            "maturity": maturity,
            "mc_discount_factor": mc,
            "curve_discount_factor": mkt,
            "mc_std_error": se,
            "error_bp_of_price": 1e4 * (mc - mkt) / mkt,
        }
