"""Model monitoring: forecast-error thresholds, input drift (PSI) and beta-stability checks.

Statuses are GREEN / AMBER / RED. Thresholds are documented in docs/monitoring_plan.md and are policy
choices (assumptions), not estimated quantities.

* Forecast error: rolling 12-month MASE of the ensemble (3-month horizon) against the same MASE scale as the
  backtest. AMBER if > 0.80 (no better than the naive-drift benchmark, MASE ~0.74 in backtest),
  RED if > 1.00 (worse than the random walk).
* Input drift: Population Stability Index of each driver between the training period (<= 2015) and the
  recent window (last 36 months). Conventional bands: PSI < 0.10 GREEN, 0.10-0.25 AMBER, > 0.25 RED.
* Beta stability: relative change of the latest filtered Kalman beta vs its 36-month-ago value (> 50% AMBER,
  > 100% RED) for each product.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from . import PROCESSED
from .deposits.features import build_frame
from .results import update_summary


def psi(expected: np.ndarray[Any, Any], actual: np.ndarray[Any, Any], bins: int = 10) -> float:
    """Population Stability Index with quantile bins from the expected (training) sample."""
    e, a = np.asarray(expected, dtype=float), np.asarray(actual, dtype=float)
    e, a = e[np.isfinite(e)], a[np.isfinite(a)]
    edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    pe = np.histogram(e, edges)[0] / len(e)
    pa = np.histogram(a, edges)[0] / len(a)
    pe, pa = np.clip(pe, 1e-4, None), np.clip(pa, 1e-4, None)
    return float(np.sum((pa - pe) * np.log(pa / pe)))


def status(value: float, amber: float, red: float) -> str:
    return "RED" if value > red else "AMBER" if value > amber else "GREEN"


def run() -> dict[str, Any]:
    df = build_frame()
    train, recent = df.loc[:"2015-12-01"], df.iloc[-36:]
    drivers = ["g", "g_m12", "dff3", "dff12", "slope", "ff"]
    psis = {c: psi(train[c].dropna().values, recent[c].dropna().values) for c in drivers}
    drift = {c: {"psi": v, "status": status(v, 0.10, 0.25)} for c, v in psis.items()}

    pr = pd.read_parquet(PROCESSED / "deposit_backtest_preds.parquet")
    tr = pd.read_parquet(PROCESSED / "deposit_truth.parquet").set_index("date")
    e = pr[pr.model == "ensemble"].set_index("origin")["3"]
    err = (e - tr["3"].reindex(e.index)).abs().dropna()
    scale = float(tr["3"].loc["2009-06-01":"2015-09-01"].abs().mean())
    roll = (err.rolling(12).mean() / scale).dropna()
    fe = {
        "latest_rolling_12m_mase_h3": float(roll.iloc[-1]),
        "as_of": str(roll.index[-1].date()),
        "status": status(float(roll.iloc[-1]), 0.80, 1.00),
        "max_rolling_mase_h3": float(roll.max()),
        "max_at": str(roll.idxmax().date()),
    }

    tv = pd.read_parquet(PROCESSED / "tv_beta.parquet").set_index("date")
    stab: dict[str, Any] = {}
    for p in ["savings_old", "money_market_old", "cd12_old", "savings_new", "cd12_new"]:
        s = tv[f"{p}|beta_filtered"].dropna()
        if len(s) > 37:
            now, then = float(s.iloc[-1]), float(s.iloc[-37])
            rel = abs(now - then) / max(abs(then), 1e-6)
            stab[p] = {"latest": now, "36m_earlier": then, "relative_change": rel, "status": status(rel, 0.5, 1.0)}
    out = {
        "forecast_error": fe,
        "input_drift": drift,
        "beta_stability": stab,
        "thresholds": {
            "mase_amber": 0.8,
            "mase_red": 1.0,
            "psi_amber": 0.10,
            "psi_red": 0.25,
            "beta_rel_amber": 0.5,
            "beta_rel_red": 1.0,
        },
        "recalibration_trigger": "any RED, or AMBER on two consecutive monthly runs",
    }
    update_summary("monitoring", out)
    return out
