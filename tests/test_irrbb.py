import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from treasury_alm.curves.nss import NSSParams
from treasury_alm.curves.zero_curve import ZeroCurve
from treasury_alm.irrbb.cashflows import CashFlows, bucketize, level_pay
from treasury_alm.irrbb.config import load_basel
from treasury_alm.irrbb.eve import EVECalculator
from treasury_alm.irrbb.outlier_test import IRRBBEngine, Overrides
from treasury_alm.irrbb.products import BaseProduct, Context
from treasury_alm.irrbb.shocks import FloorPolicy, Parallel, Short, apply_shock, flattener, standard_scenarios, steepener

BASEL = load_basel()
CURVE = ZeroCurve.from_nss(NSSParams(4.5, -1.0, 1.0, 0.5, 1.5, 7.0))
T = np.array(BASEL.midpoints)


@given(size=st.floats(1, 500), sign=st.sampled_from([-1, 1]))
def test_parallel_equal_at_all_tenors(size, sign):
    s = Parallel(size, sign).bp(T)
    assert np.allclose(s, sign * size)


@given(size=st.floats(10, 500))
def test_short_shock_decays_to_zero_and_is_monotone(size):
    s = Short(size).bp(T)
    assert abs(s[-1]) < 0.01 * size and np.all(np.diff(np.abs(s)) <= 1e-12)
    assert s[0] > 0.99 * size


@given(mode=st.sampled_from(["none", "zero", "maturity_dependent"]), size=st.floats(10, 800), sign=st.sampled_from([-1, 1]),
       base=st.floats(-0.5, 8))
@settings(max_examples=60)
def test_floor_respected(mode, size, sign, base):
    fl = FloorPolicy(mode, -1.0, 20.0)
    z = np.full_like(T, base)
    out = apply_shock(z, T, Parallel(size, sign), fl)
    floor = fl.floor(T)
    assert np.all(out >= floor - 1e-12) or base < floor.max()  # shocked rate never below the floor
    assert np.all(out >= np.minimum(floor, z) - 1e-12)


def test_rotation_coefficients_reproduce_basel_worked_example():
    """SRP31.92: 100bp short & long shocks, t=3.5y -> steepener +25.4bp, flattener -1.6bp."""
    t = np.array([3.5])
    assert abs(steepener(100, 100).bp(t)[0] - 25.4) < 0.1
    assert abs(flattener(100, 100).bp(t)[0] - (-1.6)) < 0.1


def test_six_scenarios_and_usd_sizes():
    sc = standard_scenarios(BASEL.shock_sizes["USD"], BASEL.decay_x, BASEL.coeffs())
    assert set(sc) == {"parallel_up", "parallel_down", "steepener", "flattener", "short_up", "short_down"}
    assert sc["parallel_up"].bp(T)[0] == 200 and sc["short_up"].bp(np.array([0.0]))[0] == 300
    assert len(BASEL.midpoints) == 19


def test_bucketize_conserves_amount():
    t = np.array([0.001, 0.05, 0.9, 7.3, 30.0])
    out = bucketize(t, np.ones(5), BASEL.upper)
    assert out.sum() == 5 and out[0] == 1 and out[-1] == 1


class ZeroCoupon(BaseProduct):
    side = "asset"

    def __init__(self, bal, t):
        self.name, self.balance, self.t = "zcb", bal, t

    def cashflows(self, ctx):
        return CashFlows(np.array([self.t]), np.array([self.balance * 1.3]), np.array([0.0]))

    def interest(self, ctx):
        return np.zeros(12)


@given(t=st.sampled_from([0.625, 2.5, 7.5, 12.5]), size=st.floats(25, 400), sign=st.sampled_from([-1, 1]))
@settings(max_examples=40)
def test_zero_coupon_delta_eve_matches_analytical(t, size, sign):
    """With spread calibrated at base, PV1 = B*exp(-dz*t), so dEVE = B*(exp(-dz*t) - 1) in closed form."""
    calc = EVECalculator([ZeroCoupon(10.0, t)], BASEL.upper, BASEL.midpoints)
    base = Context(CURVE, None, FloorPolicy("none"))
    calc.calibrate(base)
    shocked = Context(CURVE, Parallel(size, sign), FloorPolicy("none"))
    d = calc.eve(shocked) - calc.eve(base)
    assert d == pytest.approx(10.0 * (np.exp(-sign * size / 1e4 * t) - 1), rel=1e-6)


def test_level_pay_principal_sums_to_balance():
    cf = level_pay(100.0, 5.0, 360, 0.1)
    assert cf.principal.sum() == pytest.approx(100.0, rel=1e-9)
    assert cf.interest.sum() > 0


@pytest.fixture(scope="module")
def engine():
    return IRRBBEngine(CURVE)


def test_base_eve_equals_book_value_of_rate_sensitive_items(engine):
    a = sum(p.balance for p in engine.products if p.side == "asset")
    li = sum(p.balance for p in engine.products if p.side == "liability")
    assert engine.eve_calc.eve(engine.base_ctx) == pytest.approx(a - li, abs=1e-6)


def test_engine_results_structure_and_signs(engine):
    r = engine.run()
    assert len(r["eve"]) == 6 and len(r["nii"]) == 2
    assert r["eve"]["parallel_up"]["delta_eve_bn"] < 0 < r["eve"]["parallel_down"]["delta_eve_bn"]
    assert r["outlier_test"]["is_outlier"] == (r["outlier_test"]["worst_loss_pct_tier1"] > 15.0)


def test_beta_zero_vs_one_changes_deposit_value():
    lo = IRRBBEngine(CURVE, ov=Overrides(beta_mult=0.0)).run()
    hi = IRRBBEngine(CURVE, ov=Overrides(beta_mult=3.0)).run()
    assert hi["nii"]["parallel_up"]["delta_nii_bn"] < lo["nii"]["parallel_up"]["delta_nii_bn"]


def test_core_caps_enforced():
    e = IRRBBEngine(CURVE, ov=Overrides(core_share_delta=0.5, core_maturity_delta=10))
    for p in e.products:
        if hasattr(p, "core_share"):
            assert p.core_share <= 0.9 + 1e-12 and p.core_avg_maturity <= 5.0 + 1e-12


def test_other_currencies_run():
    r = IRRBBEngine(CURVE, ov=Overrides(currency="INR")).run()
    assert r["eve"]["parallel_up"]["pct_of_tier1"] < IRRBBEngine(CURVE).run()["eve"]["parallel_up"]["pct_of_tier1"]
