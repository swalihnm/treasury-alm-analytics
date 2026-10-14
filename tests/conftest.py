"""Shared fixtures. `sandbox` redirects every module's output paths to a temp dir so integration tests never overwrite
the real results/summary.json, figures or processed data."""

from __future__ import annotations

import importlib
import shutil

import pytest

import treasury_alm
from treasury_alm import PROCESSED

MODULES = [
    "treasury_alm.data.pipeline",
    "treasury_alm.curves.pca",
    "treasury_alm.deposits.features",
    "treasury_alm.deposits.run_betas",
    "treasury_alm.deposits.backtest",
    "treasury_alm.deposits.anomaly",
    "treasury_alm.deposits.forecasters.lstm",
    "treasury_alm.pnl.nii_drivers",
    "treasury_alm.irrbb.config",
    "treasury_alm.simulation.nii_at_risk",
    "treasury_alm.report",
]
HAVE_DATA = (PROCESSED / "monthly.parquet").exists() and (PROCESSED / "nss_params.parquet").exists()
needs_data = pytest.mark.skipif(
    not HAVE_DATA, reason="processed data not built (run `make data curves betas backtest`)"
)


@pytest.fixture()
def sandbox(tmp_path, monkeypatch):
    proc = tmp_path / "processed"
    if PROCESSED.exists():
        shutil.copytree(PROCESSED, proc)
    else:
        proc.mkdir()
    res, fig, cfg = tmp_path / "results", tmp_path / "figures", tmp_path / "config"
    (cfg / "derived").mkdir(parents=True)
    res.mkdir()
    real = treasury_alm.RESULTS / "summary.json"
    if real.exists():
        shutil.copy(real, res / "summary.json")
    for m in MODULES:
        mod = importlib.import_module(m)
        if hasattr(mod, "PROCESSED"):
            monkeypatch.setattr(mod, "PROCESSED", proc)
        if hasattr(mod, "FIGURES"):
            monkeypatch.setattr(mod, "FIGURES", fig)
        if hasattr(mod, "CONFIG") and m.endswith("run_betas"):
            monkeypatch.setattr(mod, "CONFIG", cfg)
    monkeypatch.setattr("treasury_alm.results.RESULTS", res)
    monkeypatch.setattr("treasury_alm.data.pipeline.ROOT", tmp_path)
    monkeypatch.setattr(treasury_alm, "ROOT", tmp_path)
    return tmp_path
