"""Rate-cycle identification and cumulative deposit betas.

Deposit beta = share of a policy-rate move passed to the deposit rate. It is the single most
important behavioural assumption in a bank's NII forecast: with a low beta, rising rates widen
the margin because assets reprice faster than funding.

Cycle identification is algorithmic (zig-zag on FEDFUNDS) so the dates are reproducible and
checked against the known cycles rather than hand-picked.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm


@dataclass(frozen=True)
class Cycle:
    kind: str  # "hike" | "cut"
    start: pd.Timestamp
    end: pd.Timestamp
    start_rate: float
    end_rate: float
    ongoing: bool = False

    @property
    def label(self) -> str:
        return f"{self.kind} {self.start:%Y-%m}..{self.end:%Y-%m}"


def zigzag_pivots(s: pd.Series, threshold: float) -> list[tuple[pd.Timestamp, float, str, bool]]:
    """Zig-zag pivots (date, value, 'peak'|'trough', confirmed).

    A pivot is confirmed once the series retraces by >= `threshold` pp. The final extreme is
    appended unconfirmed (the leg is still in progress) so ongoing cycles are not lost.
    """
    piv: list[tuple[pd.Timestamp, float, str, bool]] = []
    it = iter(s.items())
    i0, v0 = next(it)
    hi, lo = (i0, float(v0)), (i0, float(v0))
    direction = 0
    for i, v in it:
        v = float(v)
        if direction == 0:
            if v > hi[1]:
                hi = (i, v)
            if v < lo[1]:
                lo = (i, v)
            if v - lo[1] >= threshold:
                piv.append((lo[0], lo[1], "trough", True))
                direction, hi = 1, (i, v)
            elif hi[1] - v >= threshold:
                piv.append((hi[0], hi[1], "peak", True))
                direction, lo = -1, (i, v)
        elif direction == 1:
            if v > hi[1]:
                hi = (i, v)
            elif hi[1] - v >= threshold:
                piv.append((hi[0], hi[1], "peak", True))
                direction, lo = -1, (i, v)
        else:
            if v < lo[1]:
                lo = (i, v)
            elif v - lo[1] >= threshold:
                piv.append((lo[0], lo[1], "trough", True))
                direction, hi = 1, (i, v)
    if direction == 1:
        piv.append((hi[0], hi[1], "peak", False))
    elif direction == -1:
        piv.append((lo[0], lo[1], "trough", False))
    return piv


def identify_cycles(ff: pd.Series, threshold: float = 0.75, tol: float = 0.15) -> list[Cycle]:
    """Hiking / cutting cycles from FEDFUNDS.

    Boundaries are tightened to the *policy move*, not the flat plateau around it: the start is
    the last month still within `tol` pp of the starting extreme, and the end is the first month
    within `tol` pp of the ending extreme. A cycle whose end pivot is unconfirmed is 'ongoing'.
    """
    ff = ff.dropna()
    piv = zigzag_pivots(ff, threshold)
    cycles: list[Cycle] = []
    for (d0, v0, t0, _), (d1, v1, _t1, conf1) in zip(piv[:-1], piv[1:]):
        seg = ff.loc[d0:d1]
        if t0 == "trough":
            start = seg[seg <= v0 + tol].index[-1]
            end = seg[seg >= v1 - tol].index[0]
            kind = "hike"
        else:
            start = seg[seg >= v0 - tol].index[-1]
            end = seg[seg <= v1 + tol].index[0]
            kind = "cut"
        if end > start:
            cycles.append(Cycle(kind, start, end, float(ff[start]), float(ff[end]), ongoing=not conf1))
    return cycles


def cycle_beta(dep: pd.Series, ff: pd.Series, c: Cycle, lag: int = 0) -> dict[str, Any] | None:
    """Cumulative beta over a cycle: (d_end - d_start) / (ff_end - ff_start).

    `lag` shifts the deposit window forward by `lag` months (deposit rates respond with delay),
    clipped to deposit-data availability. Returns None when the window is not covered.
    """
    dep = dep.dropna()
    s, e = c.start + pd.DateOffset(months=lag), c.end + pd.DateOffset(months=lag)
    if dep.index.min() > c.start or dep.index.max() < e:
        return None
    df_ = float(ff[c.end] - ff[c.start])
    if abs(df_) < 1e-9:
        return None
    dd = float(dep[e] - dep[s]) if lag else float(dep[c.end] - dep[c.start])
    return {"delta_policy_pp": df_, "delta_deposit_pp": dd, "beta": dd / df_}


def cycle_table(monthly: pd.DataFrame, cycles: list[Cycle], products: dict[str, str], lags: tuple[int, ...] = (0, 3, 6)) -> list[dict[str, Any]]:
    rows = []
    ff = monthly["fedfunds"]
    for c in cycles:
        for name, col in products.items():
            dep = monthly[col].dropna()
            for lag in lags:
                r = cycle_beta(dep, ff, c, lag)
                if r is not None:
                    rows.append({"cycle": c.label, "kind": c.kind, "ongoing": c.ongoing, "product": name,
                                 "regime": "old" if col.endswith("_old") else "new", "lag_months": lag, **r})
    return rows


def rolling_beta(dep: pd.Series, ff: pd.Series, window: int = 36, horizon: int = 6) -> pd.Series:
    """Rolling OLS slope of `horizon`-month deposit-rate changes on `horizon`-month policy changes.

    Horizon-changes let pass-through lags show up; overlapping observations mean the point
    estimate is meaningful but its naive standard error is not (used only as a visual benchmark).
    """
    df_ = pd.concat([dep.diff(horizon), ff.diff(horizon)], axis=1, keys=["d", "f"]).dropna()
    out = {}
    for i in range(window, len(df_) + 1):
        w = df_.iloc[i - window:i]
        vx = float((w["f"] ** 2).sum())
        out[w.index[-1]] = float((w["f"] * w["d"]).sum() / vx) if vx > 1e-6 else np.nan
    return pd.Series(out, name="rolling_beta")


def asymmetric_beta(dep: pd.Series, ff: pd.Series, n_lags: int = 6, maxlags: int = 4) -> dict[str, Any]:
    """Up- vs down-beta from a distributed-lag model on monthly changes.

    d(dep)_t = a + sum_i bu_i * d(ff)+_{t-i} + sum_i bd_i * d(ff)-_{t-i} + e_t.
    Cumulative up-beta = sum bu_i, down-beta = sum bd_i; Wald test of equality with HAC errors.
    """
    df_ = pd.concat([dep, ff], axis=1, keys=["d", "f"]).dropna()
    dd, df = df_["d"].diff(), df_["f"].diff()
    X = {}
    for i in range(n_lags + 1):
        X[f"up{i}"] = df.clip(lower=0).shift(i)
        X[f"dn{i}"] = df.clip(upper=0).shift(i)
    Xd = pd.DataFrame(X).assign(y=dd).dropna()
    y = Xd.pop("y")
    res = sm.OLS(y, sm.add_constant(Xd)).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    up_cols = [f"up{i}" for i in range(n_lags + 1)]
    dn_cols = [f"dn{i}" for i in range(n_lags + 1)]
    names = list(res.params.index)
    R_up = np.array([1.0 if n in up_cols else 0.0 for n in names])
    R_dn = np.array([1.0 if n in dn_cols else 0.0 for n in names])
    t_up, t_dn = res.t_test(R_up), res.t_test(R_dn)
    diff = res.t_test(R_up - R_dn)
    return {
        "n_obs": int(res.nobs), "up_beta": float(t_up.effect[0]), "up_se": float(t_up.sd[0, 0]),
        "down_beta": float(t_dn.effect[0]), "down_se": float(t_dn.sd[0, 0]),
        "diff_up_minus_down": float(diff.effect[0]), "p_value_equal": float(diff.pvalue),
        "n_up_months": int((df > 0).sum()), "n_down_months": int((df < 0).sum()),
    }


def level_dependence(dep: pd.Series, ff: pd.Series, horizon: int = 6, split: float = 1.0, maxlags: int = 6) -> dict[str, Any]:
    """Does beta rise with the rate level ('beta convexity')?

    (1) Split-sample betas of horizon-changes for policy level < split vs >= split;
    (2) interaction regression d(dep) = b0*d(ff) + b1*d(ff)*level_{t-h} + e, HAC errors.
    """
    df_ = pd.concat([dep, ff], axis=1, keys=["d", "f"]).dropna()
    z = pd.DataFrame({"dd": df_["d"].diff(horizon), "df": df_["f"].diff(horizon), "lvl": df_["f"].shift(horizon)}).dropna()
    z = z[z["df"].abs() > 0.0]
    out: dict[str, Any] = {}
    for name, sub in (("low", z[z["lvl"] < split]), ("high", z[z["lvl"] >= split])):
        if len(sub) >= 12 and float((sub["df"] ** 2).sum()) > 1e-6:
            r = sm.OLS(sub["dd"], sub[["df"]]).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
            out[f"beta_{name}_level"] = float(r.params["df"])
            out[f"n_{name}"] = int(len(sub))
    if len(z) >= 24:
        X = pd.DataFrame({"df": z["df"], "df_x_lvl": z["df"] * z["lvl"]})
        r = sm.OLS(z["dd"], X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
        out["interaction_coef"] = float(r.params["df_x_lvl"])
        out["interaction_p"] = float(r.pvalues["df_x_lvl"])
        out["beta_at_zero_level"] = float(r.params["df"])
    out["split_level_pct"] = split
    return out
