import numpy as np
import pandas as pd
import pytest

from treasury_alm.deposits import betas as B
from treasury_alm.deposits.ecm import ecm_fit, simulate_ecm
from treasury_alm.deposits.kalman_beta import fit_tv_beta


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_ecm_parameter_recovery(seed):
    """Simulate with known long-run beta and adjustment speed; estimator must recover both."""
    d, f = simulate_ecm(600, beta_lr=0.35, alpha=-0.15, gamma0=0.1, seed=seed, sigma=0.02)
    r = ecm_fit(d, f)
    assert abs(r["beta_lr"] - 0.35) < 0.05
    assert abs(r["alpha"] - (-0.15)) < 0.06
    assert r["beta_lr_ci95"][0] < 0.35 < r["beta_lr_ci95"][1] or abs(r["beta_lr"] - 0.35) < 0.03


def test_kalman_recovers_constant_beta():
    rng = np.random.default_rng(5)
    n = 300
    idx = pd.date_range("2000-01-01", periods=n, freq="MS")
    f = pd.Series(np.cumsum(rng.normal(0, 0.1, n)) + 3, idx)
    x = f.diff().shift(1).rolling(6).mean()
    d = pd.Series(np.nancumsum(0.3 * x.fillna(0).values) + rng.normal(0, 0.005, n).cumsum() * 0, idx)
    d = d + rng.normal(0, 0.01, n)
    r = fit_tv_beta(d, f)
    assert abs(r["smoothed"].median() - 0.3) < 0.1


def test_zigzag_finds_cycles():
    idx = pd.date_range("2010-01-01", periods=120, freq="MS")
    path = np.r_[np.zeros(20), np.linspace(0, 3, 30), np.full(20, 3.0), np.linspace(3, 0.5, 30), np.full(20, 0.5)]
    cyc = B.identify_cycles(pd.Series(path, idx))
    assert [c.kind for c in cyc] == ["hike", "cut"]
    assert cyc[0].end_rate > 2.8 and cyc[1].end_rate < 0.7


def test_cycle_beta_arithmetic():
    idx = pd.date_range("2010-01-01", periods=40, freq="MS")
    ff = pd.Series(np.r_[np.zeros(10), np.linspace(0, 4, 10), np.full(20, 4.0)], idx)
    dep = 0.25 * ff
    c = B.identify_cycles(ff)[0]
    r = B.cycle_beta(dep, ff, c)
    assert r is not None and abs(r["beta"] - 0.25) < 1e-9


def test_asymmetric_beta_detects_asymmetry():
    rng = np.random.default_rng(3)
    n = 400
    idx = pd.date_range("1990-01-01", periods=n, freq="MS")
    df = rng.normal(0, 0.1, n)
    f = pd.Series(np.cumsum(df), idx)
    d = pd.Series(np.cumsum(0.5 * np.clip(df, 0, None) + 0.1 * np.clip(df, None, 0) + rng.normal(0, 0.005, n)), idx)
    r = B.asymmetric_beta(d, f, n_lags=1)
    assert abs(r["up_beta"] - 0.5) < 0.1 and abs(r["down_beta"] - 0.1) < 0.1 and r["p_value_equal"] < 0.01
