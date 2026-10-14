import numpy as np
import pandas as pd
import pytest

from treasury_alm.deposits.anomaly import cusum, episodes, robust_z
from treasury_alm.deposits.backtest import ensemble_from_warmup
from treasury_alm.deposits.features import HORIZONS, build_frame, targets
from treasury_alm.deposits.forecasters.base import conformal_quantile
from treasury_alm.deposits.forecasters.classical import LightGBMForecaster, NaiveForecaster, SarimaxForecaster


@pytest.fixture(scope="module")
def frame():
    return build_frame()


def test_targets_alignment(frame):
    T = targets(frame)
    t = frame.index[100]
    assert abs(T.loc[t, 3] - 100 * (np.log(frame["D"].iloc[103]) - np.log(frame["D"].iloc[100]))) < 1e-9
    assert T[12].iloc[-12:].isna().all()


@pytest.mark.parametrize("model", [NaiveForecaster("drift"), LightGBMForecaster(), SarimaxForecaster()])
def test_no_lookahead(frame, model):
    """Predictions at an origin must not change if all later rows are removed."""
    cutoff = pd.Timestamp(2015, 12, 1)
    origin = pd.DatetimeIndex([pd.Timestamp(2016, 6, 1)])
    model.fit(frame, cutoff)
    full = model.predict(frame, origin)
    trunc = model.predict(frame.loc[:origin[0]], origin)
    np.testing.assert_allclose(full.values, trunc.values, rtol=1e-8, atol=1e-8)


def test_fit_ignores_data_after_cutoff(frame):
    cutoff = pd.Timestamp(2015, 12, 1)
    origin = pd.DatetimeIndex([pd.Timestamp(2016, 6, 1)])
    a = LightGBMForecaster().fit(frame, cutoff).predict(frame, origin)
    b = LightGBMForecaster().fit(frame.loc[:cutoff], cutoff).predict(frame, origin)
    np.testing.assert_allclose(a.values, b.values)


def test_conformal_quantile_coverage():
    rng = np.random.default_rng(0)
    cover = []
    for _ in range(2000):
        calib = np.abs(rng.normal(size=39))
        y = abs(rng.normal())
        cover.append(y <= conformal_quantile(calib, 0.8))
    assert abs(np.mean(cover) - 0.8) < 0.03
    assert conformal_quantile(np.array([1.0, 2.0]), 0.95) == float("inf")


def test_ensemble_weights_sum_to_one(frame):
    T = targets(frame)
    idx = frame.loc["2012-01-01":"2017-12-01"].index
    rng = np.random.default_rng(1)
    preds = {c: pd.DataFrame({h: T.loc[idx, h] + rng.normal(0, s, len(idx)) for h in HORIZONS}) for c, s in (("a", 0.2), ("b", 0.5))}
    ens = ensemble_from_warmup(preds, T, ["a", "b"])
    for h in HORIZONS:
        w = ens.attrs["weights"][h]
        assert abs(sum(w.values()) - 1) < 1e-9 and w["a"] > w["b"]


def test_robust_z_flags_spike_and_cusum_detects_shift():
    rng = np.random.default_rng(2)
    idx = pd.date_range("2000-01-05", periods=400, freq="W-WED")
    x = pd.Series(rng.normal(0.1, 0.1, 400), idx)
    x.iloc[300] += 3.0
    z = robust_z(x)
    assert z.abs().iloc[300] > 10 and (z.abs().drop(z.index[300]) > 6).sum() == 0
    y = pd.Series(rng.normal(0, 1, 300), idx[:300])
    y.iloc[200:] += 2.0
    c = cusum(y)
    assert c.iloc[200:215].any()  # a +2 sigma shift is caught within ~15 weeks
    assert c.iloc[:200].sum() <= 1  # in-control ARL is a few hundred weeks


def test_episodes_cluster():
    idx = pd.date_range("2010-01-06", periods=30, freq="W-WED")
    f = pd.Series(False, idx)
    f.iloc[[2, 3, 4, 20]] = True
    assert len(episodes(f)) == 2
