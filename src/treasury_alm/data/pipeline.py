"""`make data`: build validated parquet tables from FRED and write the data-quality report."""

from __future__ import annotations

import json
from datetime import date
from typing import Any

import numpy as np
import polars as pl
import yaml

from .. import CONFIG, PROCESSED, RAW, ROOT
from ..results import update_summary
from . import transforms as T
from .fred_client import FredClient
from .validation import check_monotonic, gap_report, positive_schema, rate_schema

MONTHLY_RATES = {
    "fedfunds": "FEDFUNDS",
    "gs1m": "GS1M",
    "tb3ms": "TB3MS",
    "gs1": "GS1",
    "gs2": "GS2",
    "gs3": "GS3",
    "gs5": "GS5",
    "gs7": "GS7",
    "gs10": "GS10",
    "gs20": "GS20",
    "gs30": "GS30",
}
DEP_OLD = {"sav_old": "SAVNRNJ", "mm_old": "MMNRNJ", "cd12_old": "CD12NRNJ"}
DEP_NEW = {"sav_new": "SNDR", "cd12_new": "NDR12MCD"}


def fetch_all(force: bool = False) -> None:
    cfg = yaml.safe_load((CONFIG / "series.yaml").read_text())
    FredClient().fetch_many([i for v in cfg.values() for i in v], force=force)


def build_monthly() -> pl.DataFrame:
    fr: dict[str, pl.LazyFrame] = {}
    for n, s in MONTHLY_RATES.items():
        fr[n] = T.read_raw(s)
    for n, s in DEP_OLD.items():  # weekly -> monthly average
        fr[n] = T.to_monthly(T.read_raw(s), "mean")
    for n, s in DEP_NEW.items():
        fr[n] = T.read_raw(s)
    fr["deposits"] = T.read_raw("DPSACBM027SBOG")
    fr["dep_large"] = T.to_monthly(T.read_raw("DPSLCBW027SBOG"), "mean")
    fr["dep_small"] = T.to_monthly(T.read_raw("DPSSCBW027SBOG"), "mean")
    return T.wide(fr)


def build_weekly() -> pl.DataFrame:
    return T.wide(
        {
            "dep_all": T.read_raw("DPSACBW027SBOG"),
            "dep_large": T.read_raw("DPSLCBW027SBOG"),
            "dep_small": T.read_raw("DPSSCBW027SBOG"),
        }
    )


def build_daily() -> pl.DataFrame:
    return T.wide(
        {
            "sofr": T.read_raw("SOFR"),
            "dgs3mo": T.read_raw("DGS3MO"),
            "dgs2": T.read_raw("DGS2"),
            "dgs10": T.read_raw("DGS10"),
        }
    )


def build_quarterly(monthly: pl.DataFrame) -> pl.DataFrame:
    q = T.wide(
        {
            "nii": T.read_raw("QBPQYNTIY"),
            "int_income": T.read_raw("QBPQYTIY"),
            "assets": T.read_raw("QBPBSTAS"),
            "usnim": T.read_raw("USNIM"),
        }
    )
    q = q.with_columns((pl.col("int_income") - pl.col("nii")).alias("int_expense"))
    mq = (
        monthly.lazy()
        .with_columns(pl.col("date").dt.truncate("1q").alias("q"))
        .group_by("q")
        .agg(pl.col("fedfunds").mean(), pl.col("gs10").mean(), pl.col("tb3ms").mean(), pl.col("gs2").mean())
        .rename({"q": "date"})
        .collect()
    )
    return q.join(mq, on="date", how="left").sort("date")


def qbp_is_flow_check(q: pl.DataFrame) -> dict[str, Any]:
    """QBP income could be year-to-date (sawtooth: Q4 ~ 4x Q1 within each year) or quarterly flow.

    Test: mean of Q4/Q1 NII ratio within calendar years. YTD data gives ~4; flows give ~1.
    """
    d = (
        q.select("date", "nii")
        .drop_nulls()
        .with_columns(pl.col("date").dt.year().alias("y"), pl.col("date").dt.month().alias("m"))
    )
    r = []
    for y in d["y"].unique().to_list():
        a = d.filter((pl.col("y") == y) & (pl.col("m") == 1))["nii"].to_list()
        b = d.filter((pl.col("y") == y) & (pl.col("m") == 10))["nii"].to_list()
        if a and b:
            r.append(b[0] / a[0])
    m = float(np.mean(r))
    return {"mean_Q4_over_Q1": m, "n_years": len(r), "interpretation": "quarterly flow" if m < 2 else "year-to-date"}


def run(force: bool = False) -> dict[str, Any]:
    fetch_all(force)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    monthly, weekly, daily = build_monthly(), build_weekly(), build_daily()
    quarterly = build_quarterly(monthly)
    flow = qbp_is_flow_check(quarterly)
    # --- validation (raises on failure) ---
    rate_schema(list(MONTHLY_RATES) + list(DEP_OLD) + list(DEP_NEW)).validate(monthly)
    positive_schema(["deposits", "dep_large", "dep_small"]).validate(monthly)
    rate_schema(["sofr", "dgs3mo", "dgs2", "dgs10"]).validate(daily)
    positive_schema(["dep_all", "dep_large", "dep_small"]).validate(weekly)
    positive_schema(["nii", "int_income", "assets"]).validate(quarterly)
    for name, df in [("monthly", monthly), ("weekly", weekly), ("daily", daily), ("quarterly", quarterly)]:
        assert check_monotonic(df), f"{name}: dates not monotonic/unique"
        df.write_parquet(PROCESSED / f"{name}.parquet")

    meta = json.loads((RAW / "metadata.json").read_text())
    cov = {
        k: {"n_obs": v["n_obs"], "first": v["first"], "last": v["last"], "freq": v["frequency"]}
        for k, v in meta.items()
    }
    breaks = _methodology_break(monthly)
    gaps = {c: gap_report(monthly, c, 31) for c in ["deposits", "fedfunds", "gs10"]}
    gaps["sav_old(weekly)"] = gap_report(T.wide({"v": T.read_raw("SAVNRNJ")}).rename({"v": "sav"}), "sav", 7)
    nulls = {
        n: {c: int(df[c].null_count()) for c in df.columns if c != "date"}
        for n, df in [("monthly", monthly), ("quarterly", quarterly)]
    }
    vintage = date.today().isoformat()
    update_summary(
        "data",
        {
            "n_series": len(cov),
            "coverage": cov,
            "vintage_date": vintage,
            "qbp_flow_check": flow,
            "methodology_break": breaks,
            "validation": "passed",
        },
    )
    _write_report(cov, nulls, breaks, gaps, flow, vintage)
    return {"monthly": monthly.shape, "weekly": weekly.shape, "daily": daily.shape, "quarterly": quarterly.shape}


def _methodology_break(m: pl.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {"boundary": "2021-03/2021-04 (FDIC national-rate methodology change)"}
    old = m.filter(pl.col("sav_old").is_not_null()).tail(1)
    new = m.filter(pl.col("sav_new").is_not_null()).head(1)
    if old.height and new.height:
        out["savings_last_old_pct"] = float(old["sav_old"][0])
        out["savings_first_new_pct"] = float(new["sav_new"][0])
        out["fedfunds_at_boundary_pct"] = float(new["fedfunds"][0])
    return out


def _write_report(
    cov: dict[str, Any],
    nulls: dict[str, Any],
    breaks: dict[str, Any],
    gaps: dict[str, Any],
    flow: dict[str, Any],
    vintage: str,
) -> None:
    L = [
        f"# Data quality report\n\nVintage (download date): **{vintage}**. Generated by `make data`.\n",
        "## Coverage per series\n",
        "| Series | Freq | Obs | First | Last |",
        "|---|---|---|---|---|",
    ]
    for k, v in cov.items():
        L.append(f"| {k} | {v['freq']} | {v['n_obs']} | {v['first']} | {v['last']} |")
    L += [
        "\n## Validation\n",
        "- Schema: pandera (Polars). Rates within [-1%, 25%]; balances > 0; dates unique and sorted: **passed**.",
        "\n## Missing values (processed tables)\n",
    ]
    for t, d in nulls.items():
        L.append(f"- **{t}**: " + ", ".join(f"{c}={n}" for c, n in d.items() if n))
    L += [
        "\n## Detected breaks / methodology changes\n",
        f"- Deposit-rate regime boundary: {breaks['boundary']}. Old weekly series (SAVNRNJ, MMNRNJ, CD12NRNJ) end 2021-03; "
        "new monthly series (SNDR, NDR12MCD) start 2021-04. They are **kept as separate columns** and never spliced.",
        f"- Level at boundary: savings last old = {breaks.get('savings_last_old_pct')}%, first new = "
        f"{breaks.get('savings_first_new_pct'):.3f}% (Fed funds {breaks.get('fedfunds_at_boundary_pct')}%). Levels are close only because "
        "the policy rate was near zero; the FDIC changed panel/methodology, so betas are estimated within each regime and "
        "the new regime (2021-04 onwards) is the only one that covers a full hiking cycle.",
        "\n## Gaps (consecutive observations > 1.6x nominal spacing)\n",
    ]
    for k, g in gaps.items():
        L.append(f"- {k}: {g if g else 'none'}")
    L += [
        "\n## QBP income: flow vs year-to-date\n",
        f"Mean Q4/Q1 net-interest-income ratio within year = {flow['mean_Q4_over_Q1']:.3f} over {flow['n_years']} years "
        f"(YTD data would give ~4). Interpretation: **{flow['interpretation']}**. No YTD de-cumulation applied.",
        "\n## Other notes\n",
        "- NDR12MCD is the '<100M' 12-month CD national rate (FDIC); not directly comparable with CD12NRNJ (non-jumbo).",
        "- DPSACBM027SBOG is seasonally adjusted; weekly deposit series are monthly-averaged for the monthly table.",
    ]
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "data_quality.md").write_text("\n".join(L), encoding="utf-8")
