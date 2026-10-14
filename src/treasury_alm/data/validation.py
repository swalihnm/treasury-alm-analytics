"""Data validation with pandera (Polars backend) plus gap / break reporting."""
from __future__ import annotations

from typing import Any

import pandera.polars as pa
import polars as pl


def rate_schema(cols: list[str]) -> pa.DataFrameSchema:
    """Rates in percent must lie in [-1, 25]; dates unique and strictly increasing."""
    columns: dict[str, Any] = {"date": pa.Column(pl.Date, unique=True, nullable=False)}
    for c in cols:
        columns[c] = pa.Column(pl.Float64, pa.Check.in_range(-1.0, 25.0), nullable=True)
    return pa.DataFrameSchema(columns, strict=False)


def positive_schema(cols: list[str]) -> pa.DataFrameSchema:
    columns: dict[str, Any] = {"date": pa.Column(pl.Date, unique=True, nullable=False)}
    for c in cols:
        columns[c] = pa.Column(pl.Float64, pa.Check.gt(0), nullable=True)
    return pa.DataFrameSchema(columns, strict=False)


def check_monotonic(df: pl.DataFrame) -> bool:
    return bool(df["date"].is_sorted(descending=False) and df["date"].is_unique().all())


def gap_report(df: pl.DataFrame, col: str, freq_days: int, tol: float = 1.6) -> list[tuple[str, str]]:
    """Return (from, to) pairs where consecutive non-null observations exceed tol x nominal spacing."""
    d = df.select("date", col).drop_nulls().sort("date")
    gaps = d.with_columns(pl.col("date").diff().dt.total_days().alias("d"), pl.col("date").shift(1).alias("prev"))
    g = gaps.filter(pl.col("d") > freq_days * tol)
    return [(str(p), str(t)) for p, t in zip(g["prev"].to_list(), g["date"].to_list())]


def detect_level_break(a_last: float, b_first: float, a_std: float) -> float:
    """Jump across a regime boundary in units of the earlier regime's std (crude break flag)."""
    return abs(b_first - a_last) / a_std if a_std > 0 else float("nan")
