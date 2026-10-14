import numpy as np
import pandas as pd

from treasury_alm.curves.nss import NSSParams, fit_nss, par_yields
from treasury_alm.curves.pca import yield_change_pca
from treasury_alm.curves.zero_curve import BASEL_MIDPOINTS, ZeroCurve

P = NSSParams(4.5, -1.5, 2.0, -1.0, 1.5, 7.0)
MATS = np.array([1 / 12, 0.25, 1, 2, 3, 5, 7, 10, 20, 30])


def test_nss_recovers_exact_curve():
    y = par_yields(P, MATS)
    p, rmse = fit_nss(MATS, y)
    assert rmse < 0.5  # bp: model-generated yields should be fit almost exactly


def test_zero_curve_reprices_par_bond_to_100():
    zc = ZeroCurve.from_nss(P)
    for T in (2, 5, 10, 30):
        c = zc.par_yield(T)
        assert abs(zc.price_par_bond(T, c) - 100.0) < 1e-8


def test_discount_factors_decreasing_for_positive_rates():
    df = ZeroCurve.from_nss(P).bucket_discount_factors()
    assert len(df) == len(BASEL_MIDPOINTS) == 19
    assert np.all(np.diff(df) < 0)


def test_pca_variance_shares_ordered_and_level_dominates():
    rng = np.random.default_rng(0)
    idx = pd.date_range("1980-01-01", periods=400, freq="MS")
    lvl = rng.normal(0, 0.3, 400).cumsum()
    cols = ["tb3ms", "gs1", "gs2", "gs3", "gs5", "gs7", "gs10"]
    df = pd.DataFrame({c: 5 + lvl + rng.normal(0, 0.02, 400) for c in cols}, index=idx)
    r = yield_change_pca(df)
    v = r["variance_share"]
    assert v[0] > 0.9 and v[0] >= v[1] >= v[2]
