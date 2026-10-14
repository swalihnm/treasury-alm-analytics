"""Module 5: deposit-flow anomaly and regime detection.

Treasury / liquidity teams watch weekly deposit flows of large vs small banks. We score the
*detrended* flow (weekly log growth minus its trailing 52-week median, i.e. what a naive
no-news forecast would miss) with four detectors and validate them against an a-priori event list.

Detectors
---------
1. Robust z-score: residual / (1.4826 * trailing-104-week MAD of past residuals). One-sided.
2. Isolation Forest on [z_large, z_small, 4-week sums]; fitted on 1985-2007 only (pre-GFC) so that
   every labelled event is out-of-sample.
3. CUSUM (k = 0.5, h = 5 sigma) on the robust z of each bank group.
4. PELT changepoints (ruptures) on the smoothed combined z — *offline*, uses the whole sample,
   so it labels regimes after the fact rather than alerting early.
5. Module-4 residuals: monthly one-step errors of the ensemble forecaster, robust-z scored.

Event list is fixed here, BEFORE looking at detector output (rationale in the comments). Only 3
events exist, so precision/recall are coarse and reported with the raw counts.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import ruptures as rpt
from sklearn.ensemble import IsolationForest

from .. import PROCESSED
from ..results import update_summary

# (name, start, end, rationale). Windows are the periods of abnormal flows documented in the
# literature / Fed H.8 commentary; chosen a priori.
EVENTS = [
    ("lehman_flight_to_safety", "2008-09-15", "2008-11-30", "Lehman failure; flight to insured deposits"),
    (
        "pandemic_inflows",
        "2020-03-11",
        "2020-06-30",
        "COVID-19 stimulus and precautionary saving: record deposit inflows",
    ),
    (
        "regional_bank_stress",
        "2023-03-08",
        "2023-04-19",
        "SVB/Signature failures: small-bank outflows, large-bank inflows",
    ),
]
EVAL_START = "2008-01-01"
Z_THRESH = 4.0
IF_TRAIN_END = "2007-12-31"
GRACE_WEEKS = 2  # a flag up to 2 weeks before a window opens still counts as detection


def weekly_flows() -> pd.DataFrame:
    w = pl.read_parquet(PROCESSED / "weekly.parquet").to_pandas().set_index("date")
    g = 100 * np.log(w[["dep_large", "dep_small"]]).diff()
    g.columns = ["large", "small"]
    return g.dropna()


def robust_z(x: pd.Series, med_win: int = 52, mad_win: int = 104) -> pd.Series:
    """One-sided robust z: uses only past data (shifted) for centre and scale."""
    centre = x.rolling(med_win).median().shift(1)
    resid = x - centre
    mad = resid.abs().rolling(mad_win).median().shift(1)
    return resid / (1.4826 * mad)


def cusum(z: pd.Series, k: float = 0.5, h: float = 5.0) -> pd.Series:
    sp = sn = 0.0
    out = []
    for v in z.fillna(0.0).values:
        sp, sn = max(0.0, sp + v - k), max(0.0, sn - v - k)
        alarm = sp > h or sn > h
        out.append(alarm)
        if alarm:
            sp = sn = 0.0
    return pd.Series(out, z.index)


def pelt_changepoints(z: pd.Series, smooth: int = 4) -> list[pd.Timestamp]:
    s = z.fillna(0.0).clip(-15, 15).rolling(smooth).mean().dropna()
    pen = 3.0 * np.log(len(s)) * float(s.var())
    bkps = rpt.Pelt(model="l2", min_size=8, jump=1).fit(s.values.reshape(-1, 1)).predict(pen=pen)
    return [s.index[b - 1] for b in bkps[:-1]]


def episodes(flags: pd.Series, gap_weeks: int = 4) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    idx = list(flags[flags].index)
    eps: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for d in idx:
        if eps and (d - eps[-1][1]).days <= 7 * gap_weeks:
            eps[-1] = (eps[-1][0], d)
        else:
            eps.append((d, d))
    return eps


def evaluate(flags: pd.Series, name: str) -> dict[str, Any]:
    f = flags.loc[EVAL_START:]
    wins = [(n, pd.Timestamp(a), pd.Timestamp(b)) for n, a, b, _ in EVENTS]
    grace = pd.Timedelta(weeks=GRACE_WEEKS)
    det: dict[str, Any] = {}
    for n, a, b in wins:
        hit = f.loc[a - grace : b]
        first = hit[hit].index.min() if hit.any() else None
        det[n] = {
            "detected": bool(hit.any()),
            "n_flagged_weeks": int(hit.sum()),
            "first_flag": None if first is None else str(first.date()),
            "lead_lag_weeks": None if first is None else int((first - a).days / 7),
        }
    in_event = pd.Series(False, f.index)
    near_event = pd.Series(False, f.index)
    for _, a, b in wins:
        in_event |= (f.index >= a - grace) & (f.index <= b)
        near_event |= (f.index >= a - pd.Timedelta(weeks=8)) & (f.index <= b + pd.Timedelta(weeks=8))
    eps = episodes(f)
    tp = sum(any((e[1] >= a - grace) and (e[0] <= b) for _, a, b in wins) for e in eps)
    calm = f[~near_event]
    years_calm = len(calm) / 52.0
    return {
        "detector": name,
        "events": det,
        "n_events": len(wins),
        "events_detected": int(sum(d["detected"] for d in det.values())),
        "recall": float(np.mean([d["detected"] for d in det.values()])),
        "n_episodes": len(eps),
        "n_episodes_in_events": int(tp),
        "precision_episodes": float(tp / len(eps)) if eps else float("nan"),
        "false_alarm_weeks_per_year_calm": float(calm.sum() / years_calm),
        "calm_years": float(years_calm),
        "flagged_weeks_total": int(f.sum()),
    }


def run() -> dict[str, Any]:
    g = weekly_flows()
    z = pd.DataFrame({c: robust_z(g[c]) for c in g.columns}).dropna()
    zmax = z.abs().max(axis=1)
    flags = {"robust_z": zmax > Z_THRESH}
    # Isolation forest (train pre-2008)
    feat = pd.concat([z, z.rolling(4).sum().add_suffix("_4w")], axis=1).dropna()
    iso = IsolationForest(n_estimators=300, contamination=0.01, random_state=42).fit(feat.loc[:IF_TRAIN_END])
    flags["isolation_forest"] = pd.Series(iso.predict(feat) == -1, feat.index)
    flags["cusum"] = cusum(z["large"]) | cusum(z["small"])
    cps = pelt_changepoints(zmax)
    cp_flag = pd.Series(False, zmax.index)
    for c in cps:
        cp_flag.loc[c - pd.Timedelta(weeks=2) : c + pd.Timedelta(weeks=2)] = True
    flags["pelt_changepoint"] = cp_flag
    flags = {k: v.reindex(zmax.index, fill_value=False) for k, v in flags.items()}
    results = {k: evaluate(v, k) for k, v in flags.items()}
    # threshold sensitivity for the robust-z detector (so the chosen 4.0 is not cherry-picked)
    sens = {str(t): evaluate(zmax > t, f"z>{t}") for t in (3.0, 4.0, 5.0, 6.0)}
    sens = {
        t: {k: r[k] for k in ("recall", "precision_episodes", "false_alarm_weeks_per_year_calm", "n_episodes")}
        for t, r in sens.items()
    }
    # Module-4 residual layer (monthly)
    m4 = _module4_residuals()
    out = {
        "events": [{"name": n, "start": a, "end": b, "rationale": r} for n, a, b, r in EVENTS],
        "params": {
            "z_threshold": Z_THRESH,
            "eval_start": EVAL_START,
            "iforest_train_end": IF_TRAIN_END,
            "grace_weeks": GRACE_WEEKS,
        },
        "detectors": results,
        "z_threshold_sensitivity": sens,
        "pelt_changepoints": [str(c.date()) for c in cps if c >= pd.Timestamp(EVAL_START)],
        "module4_residuals": m4,
    }
    flag_df = (
        pd.DataFrame(flags)
        .assign(z_large=z["large"], z_small=z["small"], g_large=g["large"], g_small=g["small"])
        .reset_index()
    )
    flag_df.to_parquet(PROCESSED / "anomaly_flags.parquet")
    update_summary("anomalies", out)
    return out


def _module4_residuals(thr: float = 3.0) -> dict[str, Any]:
    p = pd.read_parquet(PROCESSED / "deposit_backtest_preds.parquet")
    t = pd.read_parquet(PROCESSED / "deposit_truth.parquet").set_index("date")
    e = p[p["model"] == "ensemble"].set_index("origin")["1"]
    err = (t["1"].reindex(e.index) - e).dropna()  # realised minus forecast, 1-month horizon
    z = (err - err.rolling(36).median().shift(1)) / (
        1.4826 * err.sub(err.rolling(36).median().shift(1)).abs().rolling(36).median().shift(1)
    )
    fl = z.abs() > thr
    months = [str(d.date())[:7] for d in z[fl].index]
    return {
        "threshold": thr,
        "flagged_months": months,
        "flags_2020_03_to_06": [m for m in months if "2020-03" <= m <= "2020-06"],
        "flags_2023_03_to_04": [m for m in months if "2023-03" <= m <= "2023-04"],
        "n_months_scored": int(z.notna().sum()),
    }
