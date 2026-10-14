"""Leakage-safe feature construction for deposit-balance forecasting.

Target for horizon h (months):  r_{t,h} = 100 * (log D_{t+h} - log D_t)   [percent]
Every feature in row t uses information available at the end of month t only.

`spread` (policy rate minus savings rate = the depositor's opportunity cost) exists only from
2009 because the FDIC national-rate series start then. The old (2009-2021) and new (2021-)
savings series are spliced *for this feature only*; the break is documented in
reports/data_quality.md and the two levels are within 3bp at the boundary (policy rate ~0).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

from .. import PROCESSED

HORIZONS = (1, 3, 6, 12)
FEATURES = ["g", "g_l1", "g_l2", "g_l3", "g_m3", "g_m6", "g_m12", "g_sd12", "ff", "dff3", "dff12", "spread", "slope"]
FIRST_MONTH = "1976-07-01"  # GS2 (2-year yield) starts 1976-06, needed for curve features


def load_monthly() -> pd.DataFrame:
    m = pl.read_parquet(PROCESSED / "monthly.parquet").to_pandas().set_index("date")
    m.index = pd.DatetimeIndex(m.index, freq="MS")
    return m


def build_frame(m: pd.DataFrame | None = None) -> pd.DataFrame:
    m = load_monthly() if m is None else m
    D = m["deposits"].dropna()
    df = pd.DataFrame({"D": D})
    df["g"] = 100 * np.log(D).diff()
    for k in (1, 2, 3):
        df[f"g_l{k}"] = df["g"].shift(k)
    for w in (3, 6, 12):
        df[f"g_m{w}"] = df["g"].rolling(w).mean()
    df["g_sd12"] = df["g"].rolling(12).std()
    df["ff"] = m["fedfunds"]
    df["dff3"] = m["fedfunds"].diff(3)
    df["dff12"] = m["fedfunds"].diff(12)
    sav = m["sav_old"].combine_first(m["sav_new"])
    df["spread"] = m["fedfunds"] - sav
    df["slope"] = m["gs10"] - m["tb3ms"]
    return df.loc[FIRST_MONTH:]


def targets(df: pd.DataFrame, horizons: tuple[int, ...] = HORIZONS) -> pd.DataFrame:
    """r_{t,h} aligned on origin month t (NaN where t+h is beyond the data)."""
    lD = np.log(df["D"])
    return pd.DataFrame({h: 100 * (lD.shift(-h) - lD) for h in horizons})
