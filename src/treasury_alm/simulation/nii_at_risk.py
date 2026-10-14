"""Module 8: NII-at-Risk (Earnings-at-Risk) Monte Carlo with a dynamic balance sheet.

Pipeline per path (10,000 antithetic Hull-White paths, 36 monthly steps):
 1. simulate the short rate and derive 1M/3M/6M/2Y/5Y/10Y zero rates every month (exact HW curve);
 2. deposit balances respond to rates through an ARX(1) deposit-growth model (the Module 4 driver set:
    lagged growth, 3-month policy change, curve slope; estimated on 1976-2026) with noise;
    the bank's balances deviate from its base plan by the path's growth minus the base-path growth;
    surplus deposits sit in cash, deficits drain cash and then raise wholesale funding;
 3. deposit RATES follow Module 3 betas (up/down) with an exponential lag; term deposits and
    securities reprice as ladders; mortgages prepay along the S-curve using the path's 10Y rate;
 4. monthly NII = interest income - interest expense (reduced-form, vectorised version of the products
    in `irrbb.products`; balances otherwise constant).
EaR at level p = base-path NII - (1-p) percentile of simulated NII (a positive number is a shortfall).
The 'base path' is the deterministic forward-curve path (no shocks).
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import statsmodels.api as sm
from numpy.typing import NDArray

from .. import PROCESSED
from ..curves.nss import NSSParams
from ..deposits.features import build_frame
from ..irrbb.cashflows import level_pay
from ..irrbb.config import load_balance_sheet, load_betas
from ..results import update_summary
from .hull_white import HullWhite, calibrate_ou
from .vasicek import Vasicek

Arr = NDArray[np.float64]
TENORS = (1 / 12, 0.25, 0.5, 2.0, 5.0, 10.0)
T_MONTHS = 36
N_PATHS = 10_000
SEED = 20260930


# ----------------------------------------------------------------------------- deposit growth ARX
def fit_deposit_arx() -> dict[str, Any]:
    df = build_frame()
    X = pd.DataFrame({"g_l1": df["g"].shift(1), "dff3_l1": df["dff3"].shift(1), "slope_l1": df["slope"].shift(1)})
    d = pd.concat([df["g"].rename("y"), X], axis=1).dropna()
    res = sm.OLS(d["y"], sm.add_constant(d.drop(columns="y"))).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
    ff = pd.read_parquet(PROCESSED / "monthly.parquet")[["date", "fedfunds"]].set_index("date")["fedfunds"].dropna()
    return {
        "coef": {k: float(v) for k, v in res.params.items()},
        "se": {k: float(v) for k, v in res.bse.items()},
        "sigma": float(np.sqrt(res.mse_resid)),
        "n": int(res.nobs),
        "r2": float(res.rsquared),
        "g_last": float(df["g"].iloc[-1]),
        "ff_hist": [float(x) for x in ff.iloc[-3:].values],
        "sample": f"{d.index[0].date()}..{d.index[-1].date()}",
    }


def simulate_deposit_growth(arx: dict[str, Any], z1m: Arr, z10: Arr, z3m: Arr, eps: Arr) -> Arr:
    """Monthly deposit growth (%) g[:, m], m = 1..T, given rate paths (index 0 = today)."""
    c = arx["coef"]
    N, T = z1m.shape[0], z1m.shape[1] - 1
    ff = np.concatenate(
        [np.tile(arx["ff_hist"][:-1], (N, 1)), (arx["ff_hist"][-1] + z1m - z1m[:, [0]])], axis=1
    )  # cols: ff_-2, ff_-1, ff_0..ff_T
    slope = z10 - z3m
    g = np.zeros((N, T + 1))
    g[:, 0] = arx["g_last"]
    for m in range(1, T + 1):
        dff3_prev = ff[:, 2 + m - 1] - ff[:, 2 + m - 1 - 3] if 2 + m - 1 - 3 >= 0 else np.zeros(N)
        g[:, m] = (
            c["const"]
            + c["g_l1"] * g[:, m - 1]
            + c["dff3_l1"] * dff3_prev
            + c["slope_l1"] * slope[:, m - 1]
            + eps[:, m - 1]
        )
    return g[:, 1:]


# ----------------------------------------------------------------------------- reduced-form NII model
def _ladder_rates(old: float, new: Arr, n: int, T: int) -> Arr:
    """Portfolio rate in month m (uniform ladder of n months, runoff replaced at the then-current rate)."""
    N = new.shape[0]
    out = np.zeros((N, T))
    for m in range(1, T + 1):
        lo = max(m - n, 1)
        k_new = m - lo  # entries j = lo..m-1 reprice at new[:, j-1]
        s = new[:, lo - 1 : m - 1].sum(axis=1) if k_new > 0 else 0.0
        out[:, m - 1] = (s + (n - k_new) * old) / n
    return out


class DynamicNII:
    def __init__(self, bs: dict[str, Any], betas: dict[str, dict[str, float]], adj_months: float | None = None) -> None:
        self.bs, self.betas = bs, betas
        self.tau = adj_months if adj_months is not None else float(bs["deposit_response"]["adjustment_months"])
        self.mult = float(bs["deposit_response"]["beta_multiplier"])

    def nii(self, Z: dict[float, Arr], dev: Arr) -> Arr:
        """Monthly NII [N, T] in USD bn. Z[tenor] = zero rates (%) [N, T+1]; dev = fractional deposit deviation [N, T]."""
        a, li = self.bs["assets"], self.bs["liabilities"]
        N, T = dev.shape

        def z(ten: float) -> Arr:  # rate prevailing at the START of month m (index m-1)
            return Z[ten][:, :T]

        # --- deposit deviation funding
        dep0 = sum(
            li[k]["balance"]
            for k in ("retail_transactional", "retail_non_transactional", "wholesale_nmd", "term_deposits")
        )
        d_bal = dev * dep0
        cash0 = a["cash"]["balance"]
        cash = np.maximum(cash0 + d_bal, 0.0)
        wf_extra = np.maximum(-(cash0 + d_bal), 0.0)
        inc = cash * (z(1 / 12) + a["cash"]["spread_bp"] / 100) / 1200
        # commercial floating, 1M reset
        cm = a["commercial_floating"]
        inc += cm["balance"] * (z(1 / 12) + cm["spread_bp"] / 100) / 1200
        # securities ladders
        for s in a["securities"]:
            n = int(round(24 * s["remaining_years"]))
            ten = float(s["remaining_years"])
            inc += s["balance"] * _ladder_rates(s["coupon"], Z[ten][:, :T], n, T) / 1200
        # mortgages: prepayment-driven repricing
        m = a["mortgages"]
        cp = m["cpr"]
        sched = level_pay(1.0, m["wac"], int(round(m["remaining_years"] * 12)), 0.0).principal[:36].mean()
        y = np.full(N, float(m["wac"]))
        rates_m = np.zeros((N, T))
        for k in range(T):
            rates_m[:, k] = y
            mkt = Z[10.0][:, k] + m["market_spread_bp"] / 100
            cpr = cp["min"] + (cp["max"] - cp["min"]) / (
                1 + np.exp(-cp["steepness_per_pp"] * (m["wac"] - mkt - cp["midpoint_incentive_pp"]))
            )
            phi = np.minimum(sched + (1 - (1 - cpr) ** (1 / 12)), 1.0)
            y = (1 - phi) * y + phi * mkt
        inc += m["balance"] * rates_m / 1200
        # consumer loans (amortising + prepay), replaced at current rate + shift of the 2Y
        cn = a["consumer"]
        lp = level_pay(1.0, cn["rate"], int(round(cn["wam_years"] * 12)), cn["cpr"])
        phi_c = float(np.mean(lp.principal[:12] / (1 - np.concatenate([[0], np.cumsum(lp.principal)[:11]]))))
        y = np.full(N, float(cn["rate"]))
        rc = np.zeros((N, T))
        for k in range(T):
            rc[:, k] = y
            y = (1 - phi_c) * y + phi_c * (cn["rate"] + Z[2.0][:, k] - Z[2.0][:, 0])
        inc += cn["balance"] * rc / 1200
        # ---- liabilities
        exp_ = np.zeros((N, T))
        kappa = 1 - np.exp(-1 / max(self.tau, 1e-6))
        dev_f = 1 + dev
        for key in ("retail_transactional", "retail_non_transactional", "wholesale_nmd"):
            grp = {
                "retail_transactional": "retail_transactional",
                "retail_non_transactional": "retail_non_transactional",
                "wholesale_nmd": "wholesale",
            }[key]
            b = self.betas[grp]
            r = np.full(N, float(li[key]["rate"]))
            rr = np.zeros((N, T))
            for k in range(T):
                rr[:, k] = r
                dz = Z[1 / 12][:, k] - Z[1 / 12][:, 0]
                tgt = np.maximum(li[key]["rate"] + np.where(dz >= 0, b["up"], b["down"]) * self.mult * dz, 0.0)
                r = r + kappa * (tgt - r)
            exp_ += li[key]["balance"] * dev_f * rr / 1200
        td = li["term_deposits"]
        bcd = self.betas["term_deposits"]
        n = int(td["runoff_months"])
        dz6 = Z[0.5][:, :T] - Z[0.5][:, [0]]
        new_td = td["rate"] + np.where(dz6 >= 0, bcd["up"], bcd["down"]) * self.mult * dz6
        exp_ += td["balance"] * dev_f * _ladder_rates(td["rate"], new_td, n, T) / 1200
        wf = li["wholesale_funding"]
        reset_idx = (np.arange(T) // wf["reprice_months"]) * wf["reprice_months"]
        r_wf = Z[0.25][:, reset_idx] + wf["spread_bp"] / 100
        exp_ += (wf["balance"] + wf_extra) * r_wf / 1200
        return np.asarray(inc - exp_, dtype=float)


# ----------------------------------------------------------------------------- orchestration
def _zdict(model: Any, r: Arr) -> dict[float, Arr]:
    return {t: model.zero_rates(r, t) for t in TENORS}


def _earnings_at_risk(nii_annual: Arr, base_annual: Arr) -> dict[str, Any]:
    out = {}
    for i, lab in enumerate(["year1", "year2", "year3", "cumulative_3y"]):
        x = nii_annual[:, i]
        b = float(base_annual[0, i])
        q05, q01 = np.percentile(x, 5), np.percentile(x, 1)
        out[lab] = {
            "base_bn": b,
            "mean_bn": float(x.mean()),
            "std_bn": float(x.std()),
            "p05_bn": float(q05),
            "p01_bn": float(q01),
            "p50_bn": float(np.median(x)),
            "p95_bn": float(np.percentile(x, 95)),
            "EaR95_bn": float(b - q05),
            "EaR99_bn": float(b - q01),
            "EaR95_pct_base": float(100 * (b - q05) / b),
            "EaR99_pct_base": float(100 * (b - q01) / b),
            "prob_below_base_pct": float(100 * (x < b).mean()),
        }
    return out


def _annual(m: Arr) -> Arr:
    y = np.stack([m[:, :12].sum(1), m[:, 12:24].sum(1), m[:, 24:36].sum(1)], axis=1)
    return np.column_stack([y, y.sum(1)])


def run(n_paths: int = N_PATHS) -> dict[str, Any]:
    t0 = time.time()
    p = pl.read_parquet(PROCESSED / "nss_params.parquet").to_pandas().set_index("date")
    prm = NSSParams(*p.iloc[-1][["b0", "b1", "b2", "b3", "tau1", "tau2"]].tolist())
    zero_fn = lambda t: prm.zero(np.asarray(t, dtype=float))  # noqa: E731
    m = pl.read_parquet(PROCESSED / "monthly.parquet").to_pandas().set_index("date")
    m.index = pd.DatetimeIndex(m.index)
    bs, betas = load_balance_sheet(), load_betas()
    arx = fit_deposit_arx()
    model = DynamicNII(bs, betas)

    def simulate(label: str, ou: dict[str, Any], kind: str = "hw") -> dict[str, Any]:
        if kind == "hw":
            mdl: Any = HullWhite(ou["a"], ou["sigma"], zero_fn)
            r = mdl.simulate(n_paths, T_MONTHS, SEED)
            r_base = mdl.simulate(1, T_MONTHS, SEED, deterministic=True)
        else:
            r0 = HullWhite(ou["a"], ou["sigma"], zero_fn).alpha(np.array([0.0]))[0]
            mdl = Vasicek(ou["a"], ou["long_run_mean_pct"] / 100, ou["sigma"], float(r0))
            r = mdl.simulate(n_paths, T_MONTHS, SEED)
            r_base = Vasicek(ou["a"], ou["long_run_mean_pct"] / 100, 0.0, float(r0)).simulate(
                1, T_MONTHS, SEED, antithetic=False
            )
        Z, Zb = _zdict(mdl, r), _zdict(mdl, r_base)
        rng = np.random.default_rng(SEED + 1)
        eps = rng.normal(0, arx["sigma"], (n_paths, T_MONTHS))
        eps[1::2] = -eps[0::2]  # antithetic pairs (interleaved) for deposit noise too
        g = simulate_deposit_growth(arx, Z[1 / 12], Z[10.0], Z[0.25], eps)
        gb = simulate_deposit_growth(arx, Zb[1 / 12], Zb[10.0], Zb[0.25], np.zeros((1, T_MONTHS)))
        dev = np.exp(np.cumsum(g - gb, axis=1) / 100) - 1
        nii = model.nii(Z, dev)
        base = model.nii(Zb, np.zeros((1, T_MONTHS)))
        ann, base_ann = _annual(nii), _annual(base)
        return {
            "label": label,
            "params": ou,
            "ear": _earnings_at_risk(ann, base_ann),
            "_ann": ann,
            "_base_ann": base_ann,
            "short_rate_pct": {
                "mean_by_year": [float(100 * r[:, k * 12].mean()) for k in (1, 2, 3)],
                "p05_y3": float(100 * np.percentile(r[:, 36], 5)),
                "p95_y3": float(100 * np.percentile(r[:, 36], 95)),
                "share_negative_rates_pct": float(100 * (r < 0).mean()),
            },
            "deposit_deviation_pct_y3": {
                "mean": float(100 * dev[:, -1].mean()),
                "p05": float(100 * np.percentile(dev[:, -1], 5)),
                "p95": float(100 * np.percentile(dev[:, -1], 95)),
            },
        }

    hist = m["tb3ms"].dropna()
    cal_main = calibrate_ou(hist.loc["1990-01-01":])
    cal_alt = calibrate_ou(hist.loc["1990-01-01":"2007-12-01"])
    hw = simulate("Hull-White (1990-2026 calibration)", cal_main)
    hw_alt = simulate("Hull-White (1990-2007 calibration)", cal_alt)
    vas = simulate("Vasicek benchmark (1990-2026 calibration)", cal_main, "vasicek")
    # convergence: EaR vs number of paths (first N of the main run; interleaved antithetic so every prefix is balanced)
    ann, base_ann = hw["_ann"], hw["_base_ann"]
    conv = []
    for n in (500, 1000, 2000, 4000, 6000, 8000, n_paths):
        if n > n_paths:
            continue
        e = _earnings_at_risk(ann[:n], base_ann)
        conv.append(
            {
                "n_paths": n,
                "EaR95_year1_pct": e["year1"]["EaR95_pct_base"],
                "EaR99_year1_pct": e["year1"]["EaR99_pct_base"],
                "EaR95_cum3y_pct": e["cumulative_3y"]["EaR95_pct_base"],
                "EaR99_cum3y_pct": e["cumulative_3y"]["EaR99_pct_base"],
            }
        )
    rng = np.random.default_rng(1)
    boot = []
    for _ in range(300):
        idx = rng.integers(0, len(ann), len(ann))
        boot.append(base_ann[0, 0] - np.percentile(ann[idx, 0], 5))
    check = HullWhite(cal_main["a"], cal_main["sigma"], zero_fn).martingale_check(20000, 5.0, SEED)
    np.save(PROCESSED / "mc_nii_annual.npy", ann)
    np.save(PROCESSED / "mc_nii_base.npy", base_ann)
    pd.DataFrame(conv).to_parquet(PROCESSED / "mc_convergence.parquet")
    out = {
        "n_paths": n_paths,
        "n_months": T_MONTHS,
        "seed": SEED,
        "antithetic": True,
        "runtime_seconds": time.time() - t0,
        "calibration": {"main": cal_main, "alt_pre_gfc": cal_alt},
        "martingale_check_5y": check,
        "deposit_arx": arx,
        "hull_white": {k: v for k, v in hw.items() if not k.startswith("_")},
        "hull_white_alt_calibration": {k: v for k, v in hw_alt.items() if not k.startswith("_")},
        "vasicek_benchmark": {k: v for k, v in vas.items() if not k.startswith("_")},
        "convergence": conv,
        "ear95_year1_bootstrap_se_bn": float(np.std(boot)),
        "note": "EaR = base-path NII minus the (1-p) percentile of simulated NII; positive = shortfall. Illustrative bank.",
    }
    update_summary("monte_carlo", out)
    return out
