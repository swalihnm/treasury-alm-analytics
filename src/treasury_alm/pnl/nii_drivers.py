"""Module 6: what drives banks' net interest income (FDIC Quarterly Banking Profile aggregates).

Ratios (annualised, % of average total assets):  NII ratio = 4*NII/avgA, asset yield = 4*II/avgA,
funding cost = 4*IE/avgA, where interest expense = total interest income - NII.
QBP income is a quarterly flow (checked in the data pipeline), assets are end-of-period, so
average assets = mean of opening and closing balance.

USNIM (the discontinued FRED series) is NII / *earning* assets, so it sits above our
NII / total-assets ratio; we validate co-movement (correlation, slope) and explain the level gap.
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import statsmodels.api as sm
from scipy import stats as sps
from statsmodels.stats.diagnostic import breaks_cusumolsresid
from statsmodels.tsa.stattools import adfuller, kpss

from .. import PROCESSED
from ..results import update_summary
from .rate_volume import attribution, window_attribution


def load_quarterly() -> pd.DataFrame:
    q = pl.read_parquet(PROCESSED / "quarterly.parquet").to_pandas().set_index("date")
    q.index = pd.DatetimeIndex(q.index)
    q = q.dropna(subset=["nii", "int_income", "assets"]).copy()
    q["avg_assets"] = (q["assets"] + q["assets"].shift(1)) / 2
    q = q.dropna(subset=["avg_assets"])
    q["nii_ratio"] = 400 * q["nii"] / q["avg_assets"]
    q["asset_yield"] = 400 * q["int_income"] / q["avg_assets"]
    q["funding_cost"] = 400 * q["int_expense"] / q["avg_assets"]
    q["slope"] = q["gs10"] - q["tb3ms"]
    q["d4_ff_l1"] = q["fedfunds"].diff(4).shift(1)
    q["post2008"] = (q.index >= pd.Timestamp("2008-10-01")).astype(float)
    q["post2020"] = (q.index >= pd.Timestamp("2020-01-01")).astype(float)
    return q


def beta_proxy(q: pd.DataFrame, window: int = 12) -> pd.Series:
    """Rolling slope of 4-quarter change in funding cost on 4-quarter change in the policy rate.

    CAUTION: built from interest expense, which is a component of NII, so including it in an NII
    regression is partly mechanical; we report the model with and without it.
    """
    y, x = q["funding_cost"].diff(4), q["fedfunds"].diff(4)
    out = {}
    for i in range(window, len(q) + 1):
        xs, ys = x.iloc[i - window:i], y.iloc[i - window:i]
        if xs.notna().all() and (xs**2).sum() > 0.05:
            out[q.index[i - 1]] = float((xs * ys).sum() / (xs**2).sum())
    return pd.Series(out).clip(0, 1.5).reindex(q.index)


def validate_usnim(q: pd.DataFrame) -> dict[str, Any]:
    d = q[["usnim", "nii_ratio"]].dropna()
    ols = sm.OLS(d["usnim"], sm.add_constant(d["nii_ratio"])).fit()
    ratio = (d["nii_ratio"] / d["usnim"])
    return {"n_quarters": len(d), "first": str(d.index[0].date()), "last": str(d.index[-1].date()),
            "correlation_levels": float(d.corr().iloc[0, 1]),
            "correlation_changes_4q": float(d.diff(4).dropna().corr().iloc[0, 1]),
            "mean_usnim": float(d["usnim"].mean()), "mean_nii_over_assets": float(d["nii_ratio"].mean()),
            "mean_ratio_nii_assets_over_usnim": float(ratio.mean()),
            "slope_usnim_on_ours": float(ols.params["nii_ratio"]),
            "explanation": "USNIM divides NII by earning assets; ours divides by total assets (incl. cash-like, fixed and "
                           "non-earning assets), so ours is lower by roughly the earning-asset share."}


warnings.filterwarnings("ignore", message="The test statistic is outside")


def stationarity(q: pd.DataFrame) -> dict[str, Any]:
    out = {}
    for c in ["nii_ratio", "asset_yield", "funding_cost", "fedfunds", "slope"]:
        s = q[c].dropna()
        out[c] = {"adf_p_level": float(adfuller(s, autolag="AIC")[1]), "adf_p_diff": float(adfuller(s.diff().dropna(), autolag="AIC")[1]),
                  "kpss_p_level": float(kpss(s, regression="c", nlags="auto")[1])}
    return out


def _fit(y: pd.Series, X: pd.DataFrame, maxlags: int = 4) -> tuple[Any, pd.DataFrame]:
    d = pd.concat([y.rename("y"), X], axis=1).dropna()
    res = sm.OLS(d["y"], sm.add_constant(d.drop(columns="y"))).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return res, d


def _summ(res: Any) -> dict[str, Any]:
    ci = res.conf_int()
    return {"n": int(res.nobs), "r2": float(res.rsquared), "adj_r2": float(res.rsquared_adj),
            "coef": {k: {"b": float(res.params[k]), "ci_lo": float(ci.loc[k, 0]), "ci_hi": float(ci.loc[k, 1]), "p": float(res.pvalues[k])}
                     for k in res.params.index},
            "durbin_watson": float(sm.stats.durbin_watson(res.resid))}


def chow(y: pd.Series, X: pd.DataFrame, date: str) -> dict[str, float]:
    d = pd.concat([y.rename("y"), X], axis=1).dropna()
    Xc = sm.add_constant(d.drop(columns="y"))
    k = Xc.shape[1]
    full = sm.OLS(d["y"], Xc).fit()
    a, b = d.index < pd.Timestamp(date), d.index >= pd.Timestamp(date)
    s1 = sm.OLS(d["y"][a], Xc[a]).fit().ssr
    s2 = sm.OLS(d["y"][b], Xc[b]).fit().ssr
    n = len(d)
    F = ((full.ssr - s1 - s2) / k) / ((s1 + s2) / (n - 2 * k))
    return {"date": date, "F": float(F), "p": float(1 - sps.f.cdf(F, k, n - 2 * k))}


def run() -> dict[str, Any]:
    q = load_quarterly()
    q["beta_proxy"] = beta_proxy(q)
    q["beta_proxy_l1"] = q["beta_proxy"].shift(1)
    q["d4_slope_l1"] = q["slope"].diff(4).shift(1)
    out: dict[str, Any] = {"sample": {"first": str(q.index[0].date()), "last": str(q.index[-1].date()), "n": len(q)}}
    out["usnim_validation"] = validate_usnim(q)
    out["stationarity"] = stationarity(q)

    base = q[["fedfunds", "slope"]]
    spec_levels, d1 = _fit(q["nii_ratio"], base)
    spec_full, _ = _fit(q["nii_ratio"], q[["fedfunds", "slope", "d4_ff_l1", "beta_proxy_l1"]])
    spec_nobeta, _ = _fit(q["nii_ratio"], q[["fedfunds", "slope", "d4_ff_l1"]])
    dq = pd.DataFrame({"d_nii": q["nii_ratio"].diff(4), "d_ff": q["fedfunds"].diff(4), "d_slope": q["slope"].diff(4)})
    spec_breaks, _ = _fit(q["nii_ratio"], q[["fedfunds", "slope", "d4_ff_l1", "post2008", "post2020"]])
    spec_diff, _ = _fit(dq["d_nii"], dq[["d_ff", "d_slope"]], maxlags=6)
    out["driver_regressions"] = {"levels_ff_slope": _summ(spec_levels), "levels_with_lagged_change": _summ(spec_nobeta),
                                 "levels_with_beta_proxy": _summ(spec_full), "levels_with_break_dummies": _summ(spec_breaks),
                                 "annual_differences": _summ(spec_diff)}
    out["driver_regressions"]["note"] = ("NII ratio and the policy rate are persistent (see stationarity); the levels R2 can be "
                                         "inflated by common trends, so the 4-quarter-difference specification is the cleaner estimate of the "
                                         "short-run rate/slope sensitivity.")
    out["structural_breaks"] = {
        "chow_2008Q4": chow(q["nii_ratio"], base, "2008-10-01"),
        "chow_2020Q1": chow(q["nii_ratio"], base, "2020-01-01"),
        "cusum_ols_p": float(breaks_cusumolsresid(spec_levels.resid, ddof=3)[1]),
    }
    # attribution
    qoq, yoy = attribution(q, 1), attribution(q, 4)
    win = window_attribution(q, "2021-10-01", "2023-10-01")
    funding_beta = float((q.loc["2023-10-01", "funding_cost"] - q.loc["2021-10-01", "funding_cost"]) /
                         (q.loc["2023-10-01", "fedfunds"] - q.loc["2021-10-01", "fedfunds"]))
    asset_beta = float((q.loc["2023-10-01", "asset_yield"] - q.loc["2021-10-01", "asset_yield"]) /
                       (q.loc["2023-10-01", "fedfunds"] - q.loc["2021-10-01", "fedfunds"]))
    out["attribution_2022_23_cycle"] = {
        "window": "2021Q4 -> 2023Q4 (quarterly flows, $M)", **{k: float(v) for k, v in win.items()},
        "nii_ratio_start": float(q.loc["2021-10-01", "nii_ratio"]), "nii_ratio_end": float(q.loc["2023-10-01", "nii_ratio"]),
        "nii_ratio_peak": float(q.loc["2022-01-01":"2025-01-01", "nii_ratio"].max()),
        "nii_ratio_peak_date": str(q.loc["2022-01-01":"2025-01-01", "nii_ratio"].idxmax().date()),
        "asset_yield_start": float(q.loc["2021-10-01", "asset_yield"]), "asset_yield_end": float(q.loc["2023-10-01", "asset_yield"]),
        "funding_cost_start": float(q.loc["2021-10-01", "funding_cost"]), "funding_cost_end": float(q.loc["2023-10-01", "funding_cost"]),
        "fedfunds_start": float(q.loc["2021-10-01", "fedfunds"]), "fedfunds_end": float(q.loc["2023-10-01", "fedfunds"]),
        "industry_funding_beta_vs_ff": funding_beta, "industry_asset_yield_beta_vs_ff": asset_beta,
        "share_of_dNII_from_rate": float(win["rate"] / win["d_nii"]), "share_of_dNII_from_volume": float(win["volume"] / win["d_nii"]),
    }
    out["attribution_check"] = {"max_abs_reconstruction_error_qoq": float((qoq["d_nii"] - qoq[["volume", "rate", "mix"]].sum(axis=1)).abs().max())}
    out["latest"] = {"date": str(q.index[-1].date()), "nii_ratio": float(q["nii_ratio"].iloc[-1]), "asset_yield": float(q["asset_yield"].iloc[-1]),
                     "funding_cost": float(q["funding_cost"].iloc[-1])}
    q.reset_index().pipe(pl.from_pandas).write_parquet(PROCESSED / "nii_quarterly.parquet")
    yoy.reset_index().pipe(pl.from_pandas).write_parquet(PROCESSED / "nii_attribution_yoy.parquet")
    update_summary("nii_drivers", out)
    return out
