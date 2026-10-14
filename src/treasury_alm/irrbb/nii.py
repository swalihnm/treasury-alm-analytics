"""12-month net interest income under a constant balance sheet (SRP31 earnings measure)."""
from __future__ import annotations

from .products import BaseProduct, Context


class NIICalculator:
    def __init__(self, products: list[BaseProduct]) -> None:
        self.products = products

    def nii(self, ctx: Context) -> float:
        total = 0.0
        for p in self.products:
            i = float(p.interest(ctx).sum())
            total += i if p.side == "asset" else -i
        return total

    def breakdown(self, ctx: Context) -> dict[str, float]:
        return {p.name: float(p.interest(ctx).sum()) * (1 if p.side == "asset" else -1) for p in self.products}
