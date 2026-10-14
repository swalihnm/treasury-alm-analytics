"""Error-correction model (ECM) for deposit-rate pass-through.

Why ECM: deposit and policy rates are persistent (near-unit-root) series that share a long-run
relationship d = a + beta_LR * f. Regressing changes alone loses that anchor; regressing levels
alone ignores short-run dynamics. The ECM

    D d_t = c + alpha * d_{t-1} + delta * f_{t-1} + sum_i g_i D f_{t-i} + e_t

has long-run beta_LR = -delta/alpha and speed of adjustment alpha (< 0; half-life ln(.5)/ln(1+alpha)).
Standard errors are Newey-West (HAC); beta_LR's SE uses the delta method.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen


def ecm_fit(dep: pd.Series, ff: pd.Series, n_lags: int = 3, maxlags: int = 4) -> dict[str, Any]:
    df_ = pd.concat([dep, ff], axis=1, keys=["d", "f"]).dropna()
    X = pd.DataFrame({"d_l1": df_["d"].shift(1), "f_l1": df_["f"].shift(1)})
    for i in range(n_lags + 1):
        X[f"df_l{i}"] = df_["f"].diff().shift(i)
    y = df_["d"].diff()
    Z = pd.concat([y.rename("y"), X], axis=1).dropna()
    yv = Z.pop("y")
    res = sm.OLS(yv, sm.add_constant(Z)).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    a, dl = float(res.params["d_l1"]), float(res.params["f_l1"])
    cov = res.cov_params().loc[["d_l1", "f_l1"], ["d_l1", "f_l1"]].values
    beta_lr = -dl / a if abs(a) > 1e-12 else np.nan
    g = np.array([dl / a**2, -1.0 / a])  # gradient of -delta/alpha wrt (alpha, delta)
    se_beta = float(np.sqrt(g @ cov @ g)) if abs(a) > 1e-12 else np.nan
    hl = float(np.log(0.5) / np.log(1 + a)) if -1 < a < 0 else np.nan
    short_run = float(sum(res.params[f"df_l{i}"] for i in range(n_lags + 1)))
    return {
        "n_obs": int(res.nobs), "alpha": a, "alpha_se": float(res.bse["d_l1"]), "alpha_t": float(res.tvalues["d_l1"]),
        "beta_lr": beta_lr, "beta_lr_se": se_beta,
        "beta_lr_ci95": [beta_lr - 1.96 * se_beta, beta_lr + 1.96 * se_beta],
        "half_life_months": hl, "short_run_cum_beta": short_run, "r2": float(res.rsquared),
        "resid_autocorr_lb_p": float(sm.stats.acorr_ljungbox(res.resid, lags=[6])["lb_pvalue"].iloc[0]),
    }


def cointegration_tests(dep: pd.Series, ff: pd.Series) -> dict[str, Any]:
    """ADF on levels / differences, Engle-Granger and Johansen trace tests."""
    df_ = pd.concat([dep, ff], axis=1, keys=["d", "f"]).dropna()
    out: dict[str, Any] = {"n_obs": int(len(df_))}
    out["adf_p_dep_level"] = float(adfuller(df_["d"], regression="c", autolag="AIC")[1])
    out["adf_p_policy_level"] = float(adfuller(df_["f"], regression="c", autolag="AIC")[1])
    out["adf_p_dep_diff"] = float(adfuller(df_["d"].diff().dropna(), regression="c", autolag="AIC")[1])
    eg = coint(df_["d"], df_["f"], trend="c", autolag="aic")
    out["engle_granger_stat"], out["engle_granger_p"] = float(eg[0]), float(eg[1])
    jo = coint_johansen(df_[["d", "f"]], det_order=0, k_ar_diff=2)
    out["johansen_trace_r0"] = float(jo.lr1[0])
    out["johansen_crit95_r0"] = float(jo.cvt[0, 1])
    out["johansen_rank_ge1_at_95"] = bool(jo.lr1[0] > jo.cvt[0, 1])
    return out


def simulate_ecm(n: int, beta_lr: float, alpha: float, gamma0: float, seed: int, sigma: float = 0.02,
                 intercept: float = 0.1) -> tuple[pd.Series, pd.Series]:
    """Generate (deposit, policy) series from a known ECM (used by the parameter-recovery test).

    Policy follows a regime-switching random walk with drift (hike / hold / cut phases);
    deposit rate follows D d = alpha*(d_{t-1} - intercept - beta_lr*f_{t-1}) + gamma0*Df_t + e.
    """
    rng = np.random.default_rng(seed)
    drift = np.zeros(n)
    phase = 0
    for t in range(n):
        if t % 40 == 0:
            phase = rng.choice([-1, 0, 1])
        drift[t] = 0.05 * phase
    f = np.cumsum(drift + rng.normal(0, 0.06, n)) + 3.0
    d = np.zeros(n)
    d[0] = intercept + beta_lr * f[0]
    for t in range(1, n):
        d[t] = d[t - 1] + alpha * (d[t - 1] - intercept - beta_lr * f[t - 1]) + gamma0 * (f[t] - f[t - 1]) + rng.normal(0, sigma)
    idx = pd.date_range("2000-01-01", periods=n, freq="MS")
    return pd.Series(d, idx, name="dep"), pd.Series(f, idx, name="ff")
