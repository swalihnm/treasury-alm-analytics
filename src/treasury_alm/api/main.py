"""FastAPI service: POST /irrbb/run -> dEVE and dNII for a balance sheet, currency and scenario set."""

from __future__ import annotations

import copy
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..irrbb.config import load_balance_sheet, load_basel
from ..irrbb.outlier_test import IRRBBEngine, Overrides

app = FastAPI(
    title="Treasury ALM IRRBB API",
    version="0.1.0",
    description="Basel IRRBB standardised framework on an ILLUSTRATIVE balance sheet (not a real bank).",
)
BASEL = load_basel()
CURVE_CACHE: dict[str, Any] = {}


class IRRBBRequest(BaseModel):
    currency: str = Field(
        "USD", description="Shock-size table: USD, GBP, EUR, HKD or INR (curve is always the US NSS curve)"
    )
    scenarios: list[str] | None = Field(None, description="Subset of the six EVE scenarios; default all")
    balance_sheet: dict[str, Any] | None = Field(
        None, description="Partial override merged into config/balance_sheet.yaml"
    )
    tier1_capital: float | None = None
    beta_multiplier: float = 1.0
    core_share_delta: float = 0.0
    core_maturity_delta: float = 0.0
    cpr_scale: float = 1.0
    floor_mode: str | None = Field(None, pattern="^(none|zero|maturity_dependent)$")


def _merge(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            _merge(a[k], v)
        else:
            a[k] = v
    return a


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/irrbb/run")
def irrbb_run(req: IRRBBRequest) -> dict[str, Any]:
    if req.currency not in BASEL.shock_sizes:
        raise HTTPException(422, f"unknown currency {req.currency}; choose from {sorted(BASEL.shock_sizes)}")
    bs = copy.deepcopy(load_balance_sheet())
    if req.balance_sheet:
        _merge(bs, req.balance_sheet)
    if req.tier1_capital is not None:
        bs["meta"]["tier1_capital"] = req.tier1_capital
    try:
        eng = IRRBBEngine(
            bs=bs,
            basel=BASEL,
            ov=Overrides(
                beta_mult=req.beta_multiplier,
                core_share_delta=req.core_share_delta,
                core_maturity_delta=req.core_maturity_delta,
                cpr_scale=req.cpr_scale,
                floor_mode=req.floor_mode,
                currency=req.currency,
            ),
        )
        res = eng.run()
    except (KeyError, ValueError, TypeError, AttributeError) as e:
        raise HTTPException(422, f"invalid balance sheet or parameter: {e}") from e
    if req.scenarios:
        bad = [s for s in req.scenarios if s not in res["eve"]]
        if bad:
            raise HTTPException(422, f"unknown scenarios {bad}")
        res["eve"] = {k: v for k, v in res["eve"].items() if k in req.scenarios}
    res["disclaimer"] = "Illustrative balance sheet and assumptions; not a real institution."
    return res
