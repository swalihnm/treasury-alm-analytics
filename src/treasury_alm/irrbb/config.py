"""Loaders for the Basel parameter file, the illustrative balance sheet and the measured deposit betas."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import yaml

from .. import CONFIG, PROCESSED
from ..curves.nss import NSSParams
from ..curves.zero_curve import ZeroCurve
from .shocks import FloatArr, FloorPolicy


@dataclass
class BaselParams:
    upper: FloatArr
    midpoints: FloatArr
    shock_sizes: dict[str, dict[str, float]]
    decay_x: float
    steepener: tuple[float, float]
    flattener: tuple[float, float]
    floor: FloorPolicy
    caps: dict[str, dict[str, float]]
    outlier_pct: float

    def coeffs(self) -> dict[str, tuple[float, float]]:
        return {"steepener": self.steepener, "flattener": self.flattener}


def load_basel(path: Path = CONFIG / "irrbb_basel.yaml") -> BaselParams:
    c = yaml.safe_load(path.read_text())
    fl = c["rate_floor"]
    return BaselParams(
        upper=np.array(c["time_buckets"]["upper"], dtype=float),
        midpoints=np.array(c["time_buckets"]["midpoints"], dtype=float),
        shock_sizes=c["shock_sizes_bp"],
        decay_x=float(c["shock_shape"]["decay_x"]),
        steepener=(c["shock_shape"]["steepener"]["short"], c["shock_shape"]["steepener"]["long"]),
        flattener=(c["shock_shape"]["flattener"]["short"], c["shock_shape"]["flattener"]["long"]),
        floor=FloorPolicy(fl["mode"], fl["immediate_floor_pct"], fl["floor_end_years"]),
        caps=c["core_deposit_caps"],
        outlier_pct=float(c["outlier_test"]["threshold_pct_of_tier1"]),
    )


def load_balance_sheet(path: Path = CONFIG / "balance_sheet.yaml") -> dict[str, Any]:
    return dict(yaml.safe_load(path.read_text()))


def load_betas(path: Path = CONFIG / "derived" / "deposit_betas.yaml") -> dict[str, dict[str, float]]:
    b = yaml.safe_load(path.read_text())["betas"]
    return {k: {"up": float(v["up"]), "down": float(v["down"])} for k, v in b.items()}


def latest_curve(date: str | None = None) -> tuple[ZeroCurve, str]:
    """Zero curve from the NSS fit of the latest (or a given) month."""
    p = pl.read_parquet(PROCESSED / "nss_params.parquet").to_pandas().set_index("date")
    p.index = pd.DatetimeIndex(p.index)
    row = p.iloc[-1] if date is None else p.loc[date]
    return ZeroCurve.from_nss(NSSParams(*row[["b0", "b1", "b2", "b3", "tau1", "tau2"]].tolist())), str(row.name.date())
