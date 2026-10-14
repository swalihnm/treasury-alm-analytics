"""Banking-book products. Each product generates (a) cash flows for EVE and (b) 12 months of interest for NII
under a scenario context. All amounts are USD bn; rates are in percent.

NII convention: instantaneous shock, constant balance sheet (maturing / repricing balances are replaced
by like-for-like instruments at the shocked market rate), 12-month horizon (SRP31 NII measure).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from ..curves.zero_curve import ZeroCurve
from .cashflows import CashFlows, level_pay, uniform_runoff
from .shocks import BaseShock, FloatArr, FloorPolicy, apply_shock

OVERNIGHT = 0.0028
MONTHS = np.arange(1, 13)


@dataclass
class Context:
    """A rate environment: base zero curve + optional shock (+ floor) + deposit behaviour parameters."""

    base: ZeroCurve
    shock: BaseShock | None = None
    floor: FloorPolicy = field(default_factory=FloorPolicy)
    betas: dict[str, dict[str, float]] = field(default_factory=dict)
    beta_mult: float = 1.0
    adj_months: float = 3.0
    cpr_scale: float = 1.0

    def zero(self, t: FloatArr | float) -> FloatArr:
        ta = np.atleast_1d(np.asarray(t, dtype=float))
        z0 = self.base.zero(ta)
        return z0 if self.shock is None else apply_shock(z0, ta, self.shock, self.floor)

    def shift_bp(self, t: FloatArr | float) -> FloatArr:
        ta = np.atleast_1d(np.asarray(t, dtype=float))
        return (self.zero(ta) - self.base.zero(ta)) * 100.0

    def beta(self, group: str, shift_bp: float) -> float:
        b = self.betas.get(group, {"up": 1.0, "down": 1.0})
        return float((b["up"] if shift_bp >= 0 else b["down"]) * self.beta_mult)


class BaseProduct(ABC):
    side: str  # "asset" | "liability"
    name: str
    balance: float

    @abstractmethod
    def cashflows(self, ctx: Context) -> CashFlows: ...

    @abstractmethod
    def interest(self, ctx: Context) -> FloatArr:
        """Monthly interest over the next 12 months (USD bn)."""


@dataclass
class Cash(BaseProduct):
    name: str
    balance: float
    spread_bp: float
    side: str = "asset"

    def cashflows(self, ctx: Context) -> CashFlows:
        return CashFlows(np.array([OVERNIGHT]), np.array([self.balance]), np.array([0.0]))

    def interest(self, ctx: Context) -> FloatArr:
        rate = float(ctx.zero(1 / 12)[0]) + self.spread_bp / 100.0
        return np.full(12, self.balance * rate / 1200.0)


@dataclass
class Floating(BaseProduct):
    """Floating-rate instrument: principal slotted at next reset; interest to the reset at the current rate."""

    name: str
    balance: float
    spread_bp: float
    reprice_months: int
    side: str = "asset"

    def _current(self, ctx: Context) -> float:
        return float(ctx.base.zero(max(self.reprice_months, 1) / 12)[0]) + self.spread_bp / 100.0

    def cashflows(self, ctx: Context) -> CashFlows:
        t = self.reprice_months / 12
        return CashFlows(np.array([t]), np.array([self.balance]), np.array([self.balance * self._current(ctx) / 100 * t]))

    def interest(self, ctx: Context) -> FloatArr:
        old = self._current(ctx)
        new = float(ctx.zero(max(self.reprice_months, 1) / 12)[0]) + self.spread_bp / 100.0
        rates = np.where(MONTHS <= self.reprice_months, old, new)
        return self.balance * rates / 1200.0


@dataclass
class Ladder(BaseProduct):
    """Fixed-rate balance running off evenly over n months (securities ladders, term deposits).

    NII: runoff is replaced at `new_rate = reinvest_base + beta * shock(reinvest_tenor)`; for assets
    reinvest_base is the base zero rate at the reinvestment tenor plus a spread, for term deposits it
    is the current rate (so only the measured CD beta passes through).
    """

    name: str
    balance: float
    rate: float
    n_months: int
    reinvest_tenor: float
    side: str
    beta_group: str | None = None
    new_spread_bp: float = 0.0

    def cashflows(self, ctx: Context) -> CashFlows:
        return uniform_runoff(self.balance, self.n_months, self.rate)

    def interest(self, ctx: Context) -> FloatArr:
        shift = float(ctx.shift_bp(self.reinvest_tenor)[0])
        if self.beta_group is None:
            new = float(ctx.base.zero(self.reinvest_tenor)[0]) + self.new_spread_bp / 100.0 + shift / 100.0
        else:
            new = self.rate + ctx.beta(self.beta_group, shift) * shift / 100.0
        cum = np.minimum((MONTHS - 1) / self.n_months, 1.0)
        return self.balance * ((1 - cum) * self.rate + cum * new) / 1200.0


@dataclass
class Amortizing(BaseProduct):
    """Level-pay loan with prepayments. cpr(ctx) is constant over life under the scenario curve (documented simplification)."""

    name: str
    balance: float
    rate: float
    n_months: int
    reinvest_tenor: float
    market_spread_bp: float | None  # None -> new loans priced at current coupon + shock
    cpr_fn: object  # callable(ctx) -> float
    side: str = "asset"

    def _cpr(self, ctx: Context) -> float:
        return float(min(max(self.cpr_fn(ctx) * ctx.cpr_scale, 0.0), 0.95))  # type: ignore[operator]

    def cashflows(self, ctx: Context) -> CashFlows:
        return level_pay(self.balance, self.rate, self.n_months, self._cpr(ctx))

    def interest(self, ctx: Context) -> FloatArr:
        cf = level_pay(self.balance, self.rate, self.n_months, self._cpr(ctx))
        cum_start = np.concatenate([[0.0], np.cumsum(cf.principal[:11]) / self.balance])
        shift = float(ctx.shift_bp(self.reinvest_tenor)[0])
        if self.market_spread_bp is None:
            new = self.rate + shift / 100.0
        else:
            new = float(ctx.base.zero(self.reinvest_tenor)[0]) + self.market_spread_bp / 100.0 + shift / 100.0
        return self.balance * ((1 - cum_start) * self.rate + cum_start * new) / 1200.0


def mortgage_cpr(wac: float, spread_bp: float, cmin: float, cmax: float, k: float, m0: float) -> object:
    """S-curve prepayment: CPR rises logistically with the refinancing incentive (WAC - market mortgage rate, in pp)."""

    def f(ctx: Context) -> float:
        market = float(ctx.zero(10.0)[0]) + spread_bp / 100.0
        incentive = wac - market
        return cmin + (cmax - cmin) / (1 + np.exp(-k * (incentive - m0)))

    return f


@dataclass
class NonMaturityDeposit(BaseProduct):
    """NMD with a core / non-core split (SRP31.107-112).

    Non-core balances are slotted overnight. Core balances run off linearly over 2*M years so that
    the weighted-average maturity is M (<= Basel cap); their interest cost follows the current rate
    plus beta * (scenario shift at the cash-flow date). In the NII engine the deposit rate moves by
    beta * short-rate shift, approached exponentially with time-constant `adj_months`.
    """

    name: str
    balance: float
    rate: float
    core_share: float
    core_avg_maturity: float
    group: str
    side: str = "liability"

    def __post_init__(self) -> None:
        if not 0 <= self.core_share <= 1:
            raise ValueError("core_share outside [0,1]")

    def cashflows(self, ctx: Context) -> CashFlows:
        core, noncore = self.balance * self.core_share, self.balance * (1 - self.core_share)
        n = max(int(round(24 * self.core_avg_maturity)), 1)
        t = (np.arange(n) + 1) / 12.0
        shift = ctx.shift_bp(t)
        betas = np.array([ctx.beta(self.group, s) for s in shift])
        rate = np.maximum(self.rate + betas * shift / 100.0, 0.0)
        cf = uniform_runoff(core, n, rate)
        return CashFlows(np.concatenate([[OVERNIGHT], cf.t]), np.concatenate([[noncore], cf.principal]),
                         np.concatenate([[0.0], cf.interest]))

    def interest(self, ctx: Context) -> FloatArr:
        shift = float(ctx.shift_bp(1 / 12)[0])
        w = 1 - np.exp(-MONTHS / max(ctx.adj_months, 1e-6))
        rate = np.maximum(self.rate + ctx.beta(self.group, shift) * shift / 100.0 * w, 0.0)
        return self.balance * rate / 1200.0
