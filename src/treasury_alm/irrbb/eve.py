"""Economic value of equity (EVE) on Basel time buckets.

EVE = PV(rate-sensitive assets) - PV(rate-sensitive liabilities); each product's cash flows are slotted
into the 19 Basel buckets and discounted at the bucket midpoint on (shocked zero curve + product spread).

Spread calibration: for every product a constant spread s_i is solved at the BASE curve so that PV_i = book
value. This is the standard 'market value = book value at inception' convention: it removes the (arbitrary)
credit/liquidity margin from the base case while keeping each product's cash-flow timing and optionality,
so DeltaEVE isolates the effect of the rate shock. The same s_i is reused in every shocked scenario.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from .cashflows import bucketize
from .products import BaseProduct, Context
from .shocks import FloatArr


class EVECalculator:
    def __init__(self, products: list[BaseProduct], upper: FloatArr, midpoints: FloatArr) -> None:
        self.products, self.upper, self.mid = products, upper, midpoints
        self.spreads: dict[str, float] = {}

    def slotted(self, p: BaseProduct, ctx: Context) -> FloatArr:
        cf = p.cashflows(ctx)
        return bucketize(cf.t, cf.total, self.upper)

    def _pv(self, slot: FloatArr, ctx: Context, s: float) -> float:
        z = ctx.zero(self.mid)
        return float((slot * np.exp(-(z + s) / 100.0 * self.mid)).sum())

    def calibrate(self, base_ctx: Context) -> None:
        for p in self.products:
            slot = self.slotted(p, base_ctx)
            self.spreads[p.name] = float(brentq(lambda s: self._pv(slot, base_ctx, s) - p.balance, -60.0, 60.0, xtol=1e-10))

    def product_pv(self, p: BaseProduct, ctx: Context) -> float:
        return self._pv(self.slotted(p, ctx), ctx, self.spreads[p.name])

    def eve(self, ctx: Context) -> float:
        if not self.spreads:
            raise RuntimeError("call calibrate(base_ctx) first")
        pv = 0.0
        for p in self.products:
            v = self.product_pv(p, ctx)
            pv += v if p.side == "asset" else -v
        return pv

    def gap_table(self, ctx: Context) -> list[dict[str, float]]:
        """Repricing gap: principal repricing/maturing per bucket, assets minus liabilities (USD bn)."""
        a, l_ = np.zeros(len(self.mid)), np.zeros(len(self.mid))
        for p in self.products:
            cf = p.cashflows(ctx)
            b = bucketize(cf.t, cf.principal, self.upper)
            if p.side == "asset":
                a += b
            else:
                l_ += b
        cum = np.cumsum(a - l_)
        return [{"bucket": i + 1, "midpoint_years": float(self.mid[i]), "assets": float(a[i]), "liabilities": float(l_[i]),
                 "gap": float(a[i] - l_[i]), "cumulative_gap": float(cum[i])} for i in range(len(self.mid))]
