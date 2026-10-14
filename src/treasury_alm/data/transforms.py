"""Polars transforms: frequency alignment and month-end calendar.

Rates use period averages when going weekly/daily -> monthly; stocks (deposit balances) use the
period average too (a monthly average-of-weeks is how the Fed's monthly series is defined).
Monthly dates are stamped on the first day of the month (FRED convention); `month_end` is
provided for as-of joins.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from .. import RAW


def read_raw(series_id: str, raw_dir: Path = RAW) -> pl.LazyFrame:
    """Lazily read a raw FRED CSV as (date, value)."""
    lf = pl.scan_csv(raw_dir / f"{series_id}.csv", try_parse_dates=True)
    value_col = lf.collect_schema().names()[1]
    return (
        lf.rename({value_col: "value"})
        .with_columns(pl.col("date").cast(pl.Date), pl.col("value").cast(pl.Float64))
        .sort("date")
    )


def _agg(how: str) -> pl.Expr:
    return {"mean": pl.col("value").mean(), "last": pl.col("value").last(), "sum": pl.col("value").sum()}[how]


def to_monthly(lf: pl.LazyFrame, how: str = "mean") -> pl.LazyFrame:
    """Aggregate to calendar months stamped on the first day. how: mean | last | sum."""
    return (
        lf.with_columns(pl.col("date").dt.truncate("1mo").alias("m"))
        .group_by("m")
        .agg(_agg(how).alias("value"))
        .rename({"m": "date"})
        .sort("date")
    )


def to_quarterly(lf: pl.LazyFrame, how: str = "mean") -> pl.LazyFrame:
    return (
        lf.with_columns(pl.col("date").dt.truncate("1q").alias("q"))
        .group_by("q")
        .agg(_agg(how).alias("value"))
        .rename({"q": "date"})
        .sort("date")
    )


def month_end(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf.with_columns(pl.col("date").dt.month_end().alias("month_end"))


def wide(frames: dict[str, pl.LazyFrame]) -> pl.DataFrame:
    """Full-join named (date, value) frames into one wide table keyed by date."""
    out: pl.DataFrame | None = None
    for name, lf in frames.items():
        d = lf.select("date", pl.col("value").alias(name)).collect()
        out = d if out is None else out.join(d, on="date", how="full", coalesce=True)
    assert out is not None
    return out.sort("date")
