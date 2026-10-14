import numpy as np
import pandas as pd
import pytest

from treasury_alm.curves.nss import NSSParams
from treasury_alm.simulation.hull_white import HullWhite, calibrate_ou
from treasury_alm.simulation.nii_at_risk import _ladder_rates
from treasury_alm.simulation.vasicek import Vasicek

P = NSSParams(4.5, -1.0, 1.0, 0.5, 1.5, 7.0)
ZERO = lambda t: P.zero(np.asarray(t, dtype=float))  # noqa: E731


def test_hull_white_reprices_initial_curve():
    hw = HullWhite(0.1, 0.01, ZERO)
    for T in (2.0, 5.0):
        chk = hw.martingale_check(40000, T, seed=3)
        assert abs(chk["mc_discount_factor"] - chk["curve_discount_factor"]) < 4 * chk["mc_std_error"] + 2e-4


def test_hw_zero_rate_at_time_zero_equals_curve():
    hw = HullWhite(0.1, 0.01, ZERO)
    r = hw.simulate(10, 12, 1, deterministic=True)
    for tenor in (0.5, 2.0, 10.0):
        assert hw.zero_rates(r, tenor)[0, 0] == pytest.approx(float(ZERO(np.array([tenor]))[0]), abs=2e-3)


def test_ou_calibration_recovers_parameters():
    a, sig, b, n = 0.3, 0.012, 0.04, 4000
    rng = np.random.default_rng(0)
    dt = 1 / 12
    r = np.zeros(n)
    r[0] = b
    sd = sig * np.sqrt((1 - np.exp(-2 * a * dt)) / (2 * a))
    for k in range(n - 1):
        r[k + 1] = b + (r[k] - b) * np.exp(-a * dt) + sd * rng.standard_normal()
    est = calibrate_ou(pd.Series(100 * r, pd.date_range("1900-01-01", periods=n, freq="MS")))
    assert abs(est["a"] - a) / a < 0.35 and abs(est["sigma"] - sig) / sig < 0.05
    assert abs(est["long_run_mean_pct"] - 100 * b) < 0.5


def test_antithetic_prefix_is_balanced_and_seeded():
    hw = HullWhite(0.1, 0.01, ZERO)
    r1, r2 = hw.simulate(1000, 12, 5), hw.simulate(1000, 12, 5)
    assert np.array_equal(r1, r2)
    x = r1[:500] - hw.alpha(np.arange(13) / 12)
    assert np.abs(x.mean(axis=0)).max() < 1e-12  # antithetic pairs cancel exactly in any even prefix


def test_vasicek_zero_rate_limits():
    v = Vasicek(0.2, 0.03, 0.01, 0.05)
    r = np.full((1, 2), 0.05)
    # as tenor -> 0 the zero rate approaches the short rate
    assert v.zero_rates(r, 1e-4)[0, 0] == pytest.approx(5.0, abs=0.01)


def test_ladder_rates_matches_closed_form():
    new = np.full((1, 36), 6.0)
    out = _ladder_rates(4.0, new, 12, 24)
    assert out[0, 0] == pytest.approx(4.0)             # month 1: nothing repriced yet
    assert out[0, 6] == pytest.approx(4.0 + 2.0 * 6 / 12)
    assert out[0, 20] == pytest.approx(6.0)            # fully rolled after n months
