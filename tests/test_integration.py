"""End-to-end runs of each pipeline stage on the real processed data, in a sandbox (see conftest)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from conftest import needs_data

from treasury_alm.results import load_summary, update_summary

pytestmark = needs_data


def _summary(sandbox):
    return json.loads((sandbox / "results" / "summary.json").read_text())


def test_results_writer_cleans_numpy(sandbox):
    update_summary(
        "x", {"a": np.float64(1.5), "b": np.array([1, 2]), "c": float("nan"), "d": np.int64(3), "e": np.bool_(True)}
    )
    s = load_summary()
    assert s["x"] == {"a": 1.5, "b": [1, 2], "c": None, "d": 3, "e": True}


def test_curves_run(sandbox):
    from treasury_alm.curves.pca import run

    out = run()
    assert out["nss"]["median_rmse_bp"] < 6 and out["pca"]["cumulative_3"] > 0.95
    assert _summary(sandbox)["curves"]["nss"]["n_months_fitted"] > 250


def test_betas_run(sandbox):
    from treasury_alm.deposits.run_betas import run

    out = run()
    labels = [c["label"] for c in out["cycles"]]
    assert any(label.startswith("hike 2022-03") for label in labels) and any(
        label.startswith("hike 2015-1") for label in labels
    )
    assert (sandbox / "config" / "derived" / "deposit_betas.yaml").exists()
    assert set(out["ecm"]) >= {"savings_new|full_regime", "cd12_old|ex_zlb"}


def test_backtest_run_light(sandbox, monkeypatch):
    from treasury_alm.deposits import backtest as B
    from treasury_alm.deposits.forecasters.classical import LightGBMForecaster, NaiveForecaster

    monkeypatch.setattr(
        B, "make_models", lambda: [NaiveForecaster("rw"), NaiveForecaster("drift"), LightGBMForecaster()]
    )
    out = B.run(track=True)
    assert out["protocol"]["folds_eval"] >= 5
    assert set(out["metrics"]) == {"naive_rw", "naive_drift", "lightgbm", "ensemble"}
    for lvl in ("80", "95"):
        assert 0.4 < out["intervals"][lvl]["lightgbm"][1]["coverage"] <= 1.0
    assert (sandbox / "mlflow.db").exists()
    w = out["ensemble_weights"]
    assert all(abs(sum(v.values()) - 1) < 1e-9 for v in w.values())


def test_lstm_fit_predict_shapes(sandbox):
    from treasury_alm.deposits.features import HORIZONS, build_frame
    from treasury_alm.deposits.forecasters.lstm import LSTMForecaster

    df = build_frame()
    m = LSTMForecaster(epochs=2).fit(df, pd.Timestamp(2015, 12, 1))
    p = m.predict(df, pd.DatetimeIndex([pd.Timestamp(2016, 1, 1), pd.Timestamp(2016, 2, 1)]))
    assert list(p.columns) == list(HORIZONS) and np.isfinite(p.values).all()


def test_anomaly_run(sandbox):
    from treasury_alm.deposits import anomaly

    if not (sandbox / "processed" / "deposit_backtest_preds.parquet").exists():
        pytest.skip("needs backtest output")
    out = anomaly.run()
    assert out["detectors"]["robust_z"]["events_detected"] == 3
    assert all(d["recall"] >= 2 / 3 for d in out["detectors"].values())


def test_nii_drivers_run_and_attribution(sandbox):
    from treasury_alm.pnl.nii_drivers import run
    from treasury_alm.pnl.rate_volume import decompose

    d = decompose(100.0, 1000.0, 130.0, 1100.0)
    assert d["volume"] + d["rate"] + d["mix"] == pytest.approx(d["delta"])
    out = run()
    assert out["attribution_check"]["max_abs_reconstruction_error_qoq"] < 1e-6
    assert out["usnim_validation"]["correlation_levels"] > 0.5


def test_irrbb_run(sandbox):
    from treasury_alm.irrbb.outlier_test import run

    out = run()
    assert (
        len(out["eve"]) == 6
        and len(out["gap_table"]) == 19
        and set(out["by_currency"]) >= {"USD", "GBP", "EUR", "HKD", "INR"}
    )
    assert out["most_sensitive_assumption_eve"] in out["sensitivity_max_swing_by_assumption_pp"]


def test_monte_carlo_run_small(sandbox):
    from treasury_alm.simulation.nii_at_risk import run

    out = run(n_paths=2000)
    ear = out["hull_white"]["ear"]["year1"]
    assert ear["EaR99_bn"] >= ear["EaR95_bn"] > 0
    assert abs(out["martingale_check_5y"]["error_bp_of_price"]) < 5
    assert out["convergence"][-1]["n_paths"] == 2000


def test_report_figures(sandbox):
    from treasury_alm import report

    if (
        not (sandbox / "results" / "summary.json").exists()
        or not (sandbox / "processed" / "mc_nii_annual.npy").exists()
    ):
        pytest.skip("needs a full pipeline run first")
    report.run()
    assert len(list((sandbox / "figures").glob("*.png"))) >= 10


def test_data_pipeline_from_cache(sandbox, monkeypatch):
    from treasury_alm import RAW
    from treasury_alm.data import pipeline

    if not (RAW / "FEDFUNDS.csv").exists():
        pytest.skip("raw FRED cache absent")
    monkeypatch.setenv("FRED_API_KEY", "dummy-key-cached-run")
    shapes = pipeline.run(force=False)
    assert shapes["monthly"][0] > 1000 and shapes["quarterly"][0] > 150
    s = _summary(sandbox)["data"]
    assert s["n_series"] >= 16 and s["qbp_flow_check"]["interpretation"] == "quarterly flow"
    assert (sandbox / "reports" / "data_quality.md").exists()
