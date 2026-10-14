"""Cash-flow containers, amortisation schedules and bucket slotting."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .shocks import FloatArr


@dataclass
class CashFlows:
    """Cash flows at times t (years): principal and interest kept separate (gap table uses principal)."""

    t: FloatArr
    principal: FloatArr
    interest: FloatArr

    @property
    def total(self) -> FloatArr:
        return self.principal + self.interest


def level_pay(balance: float, rate_pct: float, n_months: int, cpr: float) -> CashFlows:
    """Monthly level-payment loan with constant CPR prepayment (SMM = 1-(1-CPR)^(1/12))."""
    r = rate_pct / 1200.0
    smm = 1 - (1 - cpr) ** (1 / 12)
    bal = balance
    prin, intr = np.zeros(n_months), np.zeros(n_months)
    for k in range(n_months):
        rem = n_months - k
        pmt = bal * r / (1 - (1 + r) ** -rem) if r > 0 else bal / rem
        i = bal * r
        sched = min(pmt - i, bal)
        pre = (bal - sched) * smm if rem > 1 else 0.0
        prin[k], intr[k] = sched + pre, i
        bal -= sched + pre
    prin[-1] += max(bal, 0.0)
    t = (np.arange(n_months) + 1) / 12.0
    return CashFlows(t, prin, intr)


def uniform_runoff(balance: float, n_months: int, rate_pct: FloatArr | float) -> CashFlows:
    """Principal repaid evenly over n months; interest at `rate_pct` (scalar or per-month array) on the outstanding balance."""
    k = np.arange(n_months)
    out = balance * (1 - k / n_months)
    rate = np.broadcast_to(np.asarray(rate_pct, dtype=float), (n_months,))
    return CashFlows((k + 1) / 12.0, np.full(n_months, balance / n_months), out * rate / 1200.0)


def bucketize(t: FloatArr, amount: FloatArr, upper: FloatArr) -> FloatArr:
    """Sum `amount` into the Basel time buckets: bucket i holds (upper[i-1], upper[i]]."""
    idx = np.searchsorted(upper + 1e-6, t, side="left")
    idx = np.minimum(idx, len(upper) - 1)
    out = np.zeros(len(upper))
    np.add.at(out, idx, amount)
    return out
