"""Bank assembly, the six-scenario run, the Basel outlier test and the sensitivity (tornado) analysis."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..curves.zero_curve import ZeroCurve
from ..results import update_summary
from .config import BaselParams, latest_curve, load_balance_sheet, load_basel, load_betas
from .eve import EVECalculator
from .nii import NIICalculator
from .products import (Amortizing, BaseProduct, Cash, Context, Floating, Ladder, NonMaturityDeposit,
                       mortgage_cpr)
from .shocks import FloorPolicy, Parallel, standard_scenarios

NMD_CAP_KEY = {"retail_transactional": "retail_transactional", "retail_non_transactional": "retail_non_transactional",
               "wholesale_nmd": "wholesale"}
BETA_GROUP = {"retail_transactional": "retail_transactional", "retail_non_transactional": "retail_non_transactional",
              "wholesale_nmd": "wholesale"}


@dataclass
class Overrides:
    """Assumption perturbations used by the sensitivity analysis."""

    beta_mult: float = 1.0
    core_share_delta: float = 0.0       # percentage points / 100, applied to every NMD, clipped to caps
    core_maturity_delta: float = 0.0    # years, clipped to caps
    cpr_scale: float = 1.0
    floor_mode: str | None = None
    adj_months: float | None = None
    currency: str = "USD"
    extra: dict[str, Any] = field(default_factory=dict)


def build_products(bs: dict[str, Any], basel: BaselParams, ov: Overrides) -> list[BaseProduct]:
    a, li = bs["assets"], bs["liabilities"]
    prods: list[BaseProduct] = [Cash("cash", a["cash"]["balance"], a["cash"]["spread_bp"])]
    for s in a["securities"]:
        prods.append(Ladder(s["name"], s["balance"], s["coupon"], int(round(24 * s["remaining_years"])), s["remaining_years"], "asset"))
    m = a["mortgages"]
    c = m["cpr"]
    prods.append(Amortizing("mortgages", m["balance"], m["wac"], int(round(m["remaining_years"] * 12)), 10.0, m["market_spread_bp"],
                            mortgage_cpr(m["wac"], m["market_spread_bp"], c["min"], c["max"], c["steepness_per_pp"], c["midpoint_incentive_pp"])))
    cm = a["commercial_floating"]
    prods.append(Floating("commercial_floating", cm["balance"], cm["spread_bp"], cm["reprice_months"]))
    cn = a["consumer"]
    prods.append(Amortizing("consumer", cn["balance"], cn["rate"], int(round(cn["wam_years"] * 12)), 2.0, None, lambda ctx, v=cn["cpr"]: v))
    for key in ("retail_transactional", "retail_non_transactional", "wholesale_nmd"):
        d, cap = li[key], basel.caps[NMD_CAP_KEY[key]]
        share = float(np.clip(d["core_share"] + ov.core_share_delta, 0.0, cap["max_core"]))
        mat = float(np.clip(d["core_avg_maturity"] + ov.core_maturity_delta, 0.25, cap["max_avg_maturity_years"]))
        prods.append(NonMaturityDeposit(key, d["balance"], d["rate"], share, mat, BETA_GROUP[key]))
    td = li["term_deposits"]
    prods.append(Ladder("term_deposits", td["balance"], td["rate"], int(td["runoff_months"]), 0.5, "liability", beta_group="term_deposits"))
    wf = li["wholesale_funding"]
    prods.append(Floating("wholesale_funding", wf["balance"], wf["spread_bp"], wf["reprice_months"], side="liability"))
    return prods


class IRRBBEngine:
    def __init__(self, curve: ZeroCurve | None = None, bs: dict[str, Any] | None = None, basel: BaselParams | None = None,
                 betas: dict[str, dict[str, float]] | None = None, ov: Overrides | None = None) -> None:
        self.ov = ov or Overrides()
        self.basel = basel or load_basel()
        self.bs = bs or load_balance_sheet()
        self.curve = curve if curve is not None else latest_curve()[0]
        self.betas = betas or load_betas()
        self.products = build_products(self.bs, self.basel, self.ov)
        floor = self.basel.floor if self.ov.floor_mode is None else FloorPolicy(self.ov.floor_mode, self.basel.floor.immediate_pct, self.basel.floor.end_years)
        adj = self.ov.adj_months if self.ov.adj_months is not None else float(self.bs["deposit_response"]["adjustment_months"])
        self.mult = float(self.bs["deposit_response"]["beta_multiplier"]) * self.ov.beta_mult
        self._kw: dict[str, Any] = dict(base=self.curve, floor=floor, betas=self.betas, beta_mult=self.mult, adj_months=adj, cpr_scale=self.ov.cpr_scale)
        self.base_ctx = Context(shock=None, **self._kw)
        self.eve_calc = EVECalculator(self.products, self.basel.upper, self.basel.midpoints)
        self.eve_calc.calibrate(self.base_ctx)
        self.nii_calc = NIICalculator(self.products)
        self.tier1 = float(self.bs["meta"]["tier1_capital"])
        sizes = self.basel.shock_sizes[self.ov.currency]
        self.scenarios = standard_scenarios(sizes, self.basel.decay_x, self.basel.coeffs())
        self.nii_scenarios = {"parallel_up": self.scenarios["parallel_up"], "parallel_down": self.scenarios["parallel_down"]}

    def ctx(self, shock: Any) -> Context:
        return Context(shock=shock, **self._kw)

    def run(self) -> dict[str, Any]:
        eve0, nii0 = self.eve_calc.eve(self.base_ctx), self.nii_calc.nii(self.base_ctx)
        eve = {}
        for n, s in self.scenarios.items():
            d = self.eve_calc.eve(self.ctx(s)) - eve0
            eve[n] = {"delta_eve_bn": d, "pct_of_tier1": 100 * d / self.tier1, "pct_of_base_eve": 100 * d / eve0}
        nii = {}
        for n, s in self.nii_scenarios.items():
            d = self.nii_calc.nii(self.ctx(s)) - nii0
            nii[n] = {"delta_nii_bn": d, "pct_of_base_nii": 100 * d / nii0, "pct_of_tier1": 100 * d / self.tier1}
        worst = min(eve, key=lambda k: eve[k]["delta_eve_bn"])
        loss = -eve[worst]["pct_of_tier1"]
        thr = self.basel.outlier_pct
        return {"currency": self.ov.currency, "base_eve_bn": eve0, "base_nii_bn": nii0, "tier1_bn": self.tier1, "eve": eve, "nii": nii,
                "worst_scenario": worst, "worst_delta_eve_pct_tier1": eve[worst]["pct_of_tier1"],
                "outlier_test": {"threshold_pct_tier1": thr, "worst_loss_pct_tier1": loss, "is_outlier": bool(loss > thr)}}


def sensitivities(curve: ZeroCurve, bs: dict[str, Any], basel: BaselParams, betas: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    """One-at-a-time assumption shocks. Reports worst-case dEVE (% Tier 1) and parallel-up dNII (% base NII)."""
    cases: list[tuple[str, str, Overrides]] = [
        ("deposit beta x0.5", "deposit beta", Overrides(beta_mult=0.5)), ("deposit beta x2.0", "deposit beta", Overrides(beta_mult=2.0)),
        ("core share -10pp", "core share", Overrides(core_share_delta=-0.10)), ("core share +10pp", "core share", Overrides(core_share_delta=0.10)),
        ("core maturity -1y", "core maturity", Overrides(core_maturity_delta=-1.0)), ("core maturity +1y", "core maturity", Overrides(core_maturity_delta=1.0)),
        ("mortgage CPR x0.5", "prepayment speed", Overrides(cpr_scale=0.5)), ("mortgage CPR x1.5", "prepayment speed", Overrides(cpr_scale=1.5)),
        ("deposit lag 1m", "deposit lag", Overrides(adj_months=1.0)), ("deposit lag 6m", "deposit lag", Overrides(adj_months=6.0)),
        ("floor: none", "rate floor", Overrides(floor_mode="none")), ("floor: zero", "rate floor", Overrides(floor_mode="zero")),
    ]
    base = IRRBBEngine(curve, bs, basel, betas).run()
    rows = []
    for label, group, ov in cases:
        r = IRRBBEngine(curve, bs, basel, betas, ov).run()
        rows.append({"case": label, "assumption": group, "worst_eve_pct_tier1": r["worst_delta_eve_pct_tier1"], "worst_scenario": r["worst_scenario"],
                     "eve_swing_vs_base_pp": r["worst_delta_eve_pct_tier1"] - base["worst_delta_eve_pct_tier1"],
                     "nii_parallel_up_pct": r["nii"]["parallel_up"]["pct_of_base_nii"], "nii_parallel_down_pct": r["nii"]["parallel_down"]["pct_of_base_nii"],
                     "nii_swing_up_pp": r["nii"]["parallel_up"]["pct_of_base_nii"] - base["nii"]["parallel_up"]["pct_of_base_nii"]})
    return rows


def run() -> dict[str, Any]:
    curve, asof = latest_curve()
    basel, bs, betas = load_basel(), load_balance_sheet(), load_betas()
    eng = IRRBBEngine(curve, bs, basel, betas)
    res = eng.run()
    res["curve_asof"] = asof
    res["label"] = bs["meta"]["label"]
    res["gap_table"] = eng.eve_calc.gap_table(eng.base_ctx)
    res["spreads_pct"] = eng.eve_calc.spreads
    res["nii_breakdown_base"] = eng.nii_calc.breakdown(eng.base_ctx)
    res["by_currency"] = {c: IRRBBEngine(curve, bs, basel, betas, Overrides(currency=c)).run() for c in basel.shock_sizes}
    sens = sensitivities(curve, bs, basel, betas)
    res["sensitivity"] = sens
    by_group: dict[str, float] = {}
    nii_group: dict[str, float] = {}
    for r in sens:
        by_group[r["assumption"]] = max(by_group.get(r["assumption"], 0.0), abs(r["eve_swing_vs_base_pp"]))
        nii_group[r["assumption"]] = max(nii_group.get(r["assumption"], 0.0), abs(r["nii_swing_up_pp"]))
    res["most_sensitive_assumption_eve"] = max(by_group, key=lambda k: by_group[k])
    res["most_sensitive_assumption_nii"] = max(nii_group, key=lambda k: nii_group[k])
    res["sensitivity_max_swing_by_assumption_pp"] = by_group
    res["sensitivity_max_nii_swing_by_assumption_pp"] = nii_group
    update_summary("irrbb", res)
    return res
