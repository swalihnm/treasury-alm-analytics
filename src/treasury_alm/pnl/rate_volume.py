"""Rate / volume / mix attribution of changes in interest income and expense.

For a P&L line X = y * A  (y = yield or funding cost per unit of average assets, A = average assets):

    X1 - X0 = (A1 - A0) * y0      volume effect : balance sheet growth at the old margin
            + (y1 - y0) * A0      rate effect   : repricing of the old balance sheet
            + (A1 - A0)(y1 - y0)  mix/joint     : interaction of the two

With only *aggregate* industry data the 'mix' term is the pure interaction (a product-level
decomposition would also need balances by product, which FRED does not provide). The three terms
sum exactly to the change, which the unit tests assert. NII = income - expense, so
dNII = (volume+rate+mix on income) - (volume+rate+mix on expense).
"""

from __future__ import annotations

import pandas as pd


def decompose(x0: float, a0: float, x1: float, a1: float) -> dict[str, float]:
    y0, y1 = x0 / a0, x1 / a1
    vol, rate, mix = (a1 - a0) * y0, (y1 - y0) * a0, (a1 - a0) * (y1 - y0)
    return {"delta": x1 - x0, "volume": vol, "rate": rate, "mix": mix}


def attribution(q: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
    """Attribute dNII over `periods` quarters (1 = QoQ, 4 = YoY) for every quarter in `q`.

    `q` needs columns int_income, int_expense (flows, $M) and avg_assets ($M).
    """
    rows = []
    for i in range(periods, len(q)):
        p0, p1 = q.iloc[i - periods], q.iloc[i]
        inc = decompose(p0["int_income"], p0["avg_assets"], p1["int_income"], p1["avg_assets"])
        exp = decompose(p0["int_expense"], p0["avg_assets"], p1["int_expense"], p1["avg_assets"])
        rows.append(
            {
                "date": q.index[i],
                "d_nii": inc["delta"] - exp["delta"],
                "volume": inc["volume"] - exp["volume"],
                "rate": inc["rate"] - exp["rate"],
                "mix": inc["mix"] - exp["mix"],
                "income_volume": inc["volume"],
                "income_rate": inc["rate"],
                "income_mix": inc["mix"],
                "expense_volume": exp["volume"],
                "expense_rate": exp["rate"],
                "expense_mix": exp["mix"],
                "d_income": inc["delta"],
                "d_expense": exp["delta"],
            }
        )
    return pd.DataFrame(rows).set_index("date")


def window_attribution(q: pd.DataFrame, start: str, end: str) -> dict[str, float]:
    """Single-window (start quarter -> end quarter) decomposition of NII, income and expense."""
    p0, p1 = q.loc[start], q.loc[end]
    inc = decompose(p0["int_income"], p0["avg_assets"], p1["int_income"], p1["avg_assets"])
    exp = decompose(p0["int_expense"], p0["avg_assets"], p1["int_expense"], p1["avg_assets"])
    out = {
        "d_nii": inc["delta"] - exp["delta"],
        "volume": inc["volume"] - exp["volume"],
        "rate": inc["rate"] - exp["rate"],
        "mix": inc["mix"] - exp["mix"],
    }
    out.update({f"income_{k}": v for k, v in inc.items()})
    out.update({f"expense_{k}": v for k, v in exp.items()})
    return out
