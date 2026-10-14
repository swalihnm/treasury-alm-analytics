"""Rolling-origin (expanding-window) backtest, metrics, conformal intervals, MLflow tracking.

Protocol
--------
* Annual folds. For fold year Y the model is fit on information up to Dec(Y-1) (targets must
  resolve by then) and forecasts every month-origin in Y using data up to that origin.
* 2012-2015 = warm-up: used ONLY to fit ensemble weights and to seed conformal calibration.
* 2016-onwards = evaluation (includes 2020 pandemic inflows and the 2022-23 hiking/outflow period).
* MASE denominator = in-sample mean |r_{t,h}| of the zero-growth random walk over 2009-06..2015-12
  (fixed, identical for every model). sMAPE is computed on deposit *levels* (growth rates near zero
  make sMAPE meaningless); RMSE on growth in percentage points.
* Conformal: for origin t, calibration set = the last 36 *resolved* out-of-sample |errors| of the
  same model and horizon (origins s with s+h <= t). Interval = point +/- the ceil((n+1)*level)-th
  smallest. No test-period information is used.
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from .. import PROCESSED
from ..results import update_summary
from .features import HORIZONS, build_frame, targets
from .forecasters.base import BaseForecaster, conformal_quantile
from .forecasters.classical import LightGBMForecaster, NaiveForecaster, SarimaxForecaster
from .forecasters.lstm import LSTMForecaster

WARMUP_START, EVAL_START = 2012, 2016
CALIB_WINDOW = 36
STRESS = {"covid_2020": ("2020-03-01", "2020-12-01"), "hiking_2022_23": ("2022-03-01", "2023-12-01")}


def make_models() -> list[BaseForecaster]:
    return [NaiveForecaster("rw"), NaiveForecaster("drift"), SarimaxForecaster(), LightGBMForecaster(), LSTMForecaster()]


def collect_predictions(df: pd.DataFrame, models: list[BaseForecaster], verbose: bool = True) -> dict[str, pd.DataFrame]:
    last = df.index[-1]
    preds: dict[str, list[pd.DataFrame]] = {m.name: [] for m in models}
    for year in range(WARMUP_START, last.year + 1):
        cutoff = pd.Timestamp(year - 1, 12, 1)
        origins = df.loc[f"{year}-01-01": f"{year}-12-01"].index
        for m in models:
            m.fit(df, cutoff)
            preds[m.name].append(m.predict(df, origins))
        if verbose:
            print(f"fold {year} done", flush=True)
    return {k: pd.concat(v) for k, v in preds.items()}


def ensemble_from_warmup(preds: dict[str, pd.DataFrame], truth: pd.DataFrame, components: list[str]) -> pd.DataFrame:
    """Inverse-squared-MAE weights per horizon from warm-up years only; applied to all origins."""
    out = pd.DataFrame(index=next(iter(preds.values())).index, columns=list(HORIZONS), dtype=float)
    weights: dict[int, dict[str, float]] = {}
    for h in HORIZONS:
        wu = [i for i in out.index if WARMUP_START <= i.year < EVAL_START and not np.isnan(truth.loc[i, h]) and i + pd.DateOffset(months=h) < pd.Timestamp(EVAL_START, 1, 1)]
        mae = {c: float(np.mean(np.abs(preds[c].loc[wu, h] - truth.loc[wu, h]))) for c in components}
        raw = {c: 1.0 / mae[c] ** 2 for c in components}
        s = sum(raw.values())
        weights[h] = {c: raw[c] / s for c in components}
        out[h] = sum(weights[h][c] * preds[c][h] for c in components)
    out.attrs["weights"] = weights
    return out


def _mase_scale(truth: pd.DataFrame) -> dict[int, float]:
    tr = truth.loc["2009-06-01":"2015-12-01"]
    return {h: float(np.nanmean(np.abs(tr[h].dropna().loc[:pd.Timestamp(2015, 12, 1) - pd.DateOffset(months=h)]))) for h in HORIZONS}


def metrics_table(preds: dict[str, pd.DataFrame], df: pd.DataFrame, truth: pd.DataFrame, start: str = "2016-01-01",
                  end: str | None = None) -> dict[str, dict[int, dict[str, float]]]:
    scale = _mase_scale(truth)
    out: dict[str, dict[int, dict[str, float]]] = {}
    for name, p in preds.items():
        out[name] = {}
        for h in HORIZONS:
            idx = [i for i in p.loc[start:end].index if not np.isnan(truth.loc[i, h])]
            err = p.loc[idx, h] - truth.loc[idx, h]
            f_lvl = df.loc[idx, "D"] * np.exp(p.loc[idx, h] / 100)
            a_lvl = df.loc[idx, "D"] * np.exp(truth.loc[idx, h] / 100)
            smape = float(np.mean(200 * np.abs(a_lvl - f_lvl) / (np.abs(a_lvl) + np.abs(f_lvl))))
            out[name][h] = {"MASE": float(np.mean(np.abs(err)) / scale[h]), "sMAPE_level_pct": smape,
                            "RMSE_pp": float(np.sqrt(np.mean(err**2))), "n": len(idx)}
    return out


def conformal_eval(p: pd.DataFrame, truth: pd.DataFrame, level: float, start: str = "2016-01-01") -> dict[int, dict[str, float]]:
    """Rolling split-conformal coverage / width / pinball for one model's predictions."""
    res: dict[int, dict[str, float]] = {}
    a = 1 - level
    for h in HORIZONS:
        cover, width, pin = [], [], []
        for t in p.loc[start:].index:
            if np.isnan(truth.loc[t, h]):
                continue
            hist = [s for s in p.index if s + pd.DateOffset(months=h) <= t and not np.isnan(truth.loc[s, h])][-CALIB_WINDOW:]
            q = conformal_quantile(np.abs(p.loc[hist, h].values - truth.loc[hist, h].values), level)
            if not np.isfinite(q):
                continue
            lo, hi, y = p.loc[t, h] - q, p.loc[t, h] + q, truth.loc[t, h]
            cover.append(lo <= y <= hi)
            width.append(hi - lo)
            tl, th = a / 2, 1 - a / 2
            pin.append(0.5 * (max(tl * (y - lo), (tl - 1) * (y - lo)) + max(th * (y - hi), (th - 1) * (y - hi))))
        res[h] = {"coverage": float(np.mean(cover)), "mean_width_pp": float(np.mean(width)), "pinball": float(np.mean(pin)), "n": len(cover)}
    return res


def run(track: bool = True) -> dict[str, Any]:
    warnings.filterwarnings("ignore")
    df = build_frame()
    truth = targets(df)
    models = make_models()
    preds = collect_predictions(df, models)
    comps = ["naive_drift", "sarimax", "lightgbm", "lstm"]
    preds["ensemble"] = ensemble_from_warmup(preds, truth, comps)
    weights = preds["ensemble"].attrs["weights"]
    table = metrics_table(preds, df, truth)
    stress = {k: metrics_table(preds, df, truth, a, b) for k, (a, b) in STRESS.items()}
    calm_end = "2019-12-01"
    stress["calm_2016_2019"] = metrics_table(preds, df, truth, "2016-01-01", calm_end)
    intervals = {f"{int(l*100)}": {n: conformal_eval(p, truth, l) for n, p in preds.items() if n != "naive_rw"} for l in (0.8, 0.95)}
    # headline: best model vs naive (mean MASE across horizons)
    mean_mase = {n: float(np.mean([table[n][h]["MASE"] for h in HORIZONS])) for n in table}
    cand = {n: v for n, v in mean_mase.items() if not n.startswith("naive")}
    best = min(cand, key=lambda k: cand[k])
    improvement = {
        "best_model": best, "mean_MASE": mean_mase,
        "vs_naive_rw_pct": {h: 100 * (1 - table[best][h]["MASE"] / table["naive_rw"][h]["MASE"]) for h in HORIZONS},
        "vs_naive_drift_pct": {h: 100 * (1 - table[best][h]["MASE"] / table["naive_drift"][h]["MASE"]) for h in HORIZONS},
        "mean_vs_naive_rw_pct": 100 * (1 - mean_mase[best] / mean_mase["naive_rw"]),
        "mean_vs_naive_drift_pct": 100 * (1 - mean_mase[best] / mean_mase["naive_drift"]),
    }
    n_folds = df.index[-1].year - EVAL_START + 1
    out = {"protocol": {"folds_eval": n_folds, "eval_start": f"{EVAL_START}-01", "warmup": f"{WARMUP_START}-{EVAL_START - 1}",
                        "last_origin": str(df.index[-1].date()), "horizons_months": list(HORIZONS), "calib_window": CALIB_WINDOW},
           "metrics": table, "stress_windows": stress, "intervals": intervals, "headline": improvement,
           "ensemble_weights": weights}
    # persist predictions for anomaly module and charts
    long = pd.concat({n: p for n, p in preds.items()}, names=["model", "origin"]).reset_index()
    long.columns = [str(c) for c in long.columns]
    pd.DataFrame(long).to_parquet(PROCESSED / "deposit_backtest_preds.parquet")
    truth.rename(columns=str).reset_index().to_parquet(PROCESSED / "deposit_truth.parquet")
    update_summary("forecasting", out)
    if track:
        _log_mlflow(table, intervals, weights)
    return out


def _log_mlflow(table: dict[str, Any], intervals: dict[str, Any], weights: dict[int, dict[str, float]]) -> None:
    import mlflow

    from .. import ROOT
    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
    mlflow.set_experiment("deposit_forecasting")
    for name, byh in table.items():
        with mlflow.start_run(run_name=name):
            mlflow.log_param("model", name)
            mlflow.log_param("eval_start", EVAL_START)
            for h, m in byh.items():
                for k, v in m.items():
                    mlflow.log_metric(f"{k}_h{h}", v)
            if name in intervals["80"]:
                for h, m in intervals["80"][name].items():
                    mlflow.log_metric(f"cov80_h{h}", m["coverage"])
                for h, m in intervals["95"][name].items():
                    mlflow.log_metric(f"cov95_h{h}", m["coverage"])
            if name == "ensemble":
                mlflow.log_dict({str(k): v for k, v in weights.items()}, "ensemble_weights.json")
