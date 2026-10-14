"""FRED client: fetch with retries, cache raw CSV and record the vintage in metadata.json.

Why: reproducibility. A model result is only defensible if we know which data vintage produced
it, so every pull stores series metadata and the download date.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
from fredapi import Fred

from .. import RAW, ROOT


def _load_key() -> str:
    key = os.environ.get("FRED_API_KEY")
    env = ROOT / ".env"
    if not key and env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("FRED_API_KEY="):
                key = line.split("=", 1)[1].strip()
    if not key:
        raise RuntimeError("FRED_API_KEY not set (environment variable or .env file)")
    return key


class FredClient:
    def __init__(self, raw_dir: Path = RAW, retries: int = 4, api_key: str | None = None) -> None:
        self.raw_dir = raw_dir
        self.retries = retries
        self._fred = Fred(api_key=api_key or _load_key())
        raw_dir.mkdir(parents=True, exist_ok=True)

    def _retry(self, fn: Callable[..., Any], *a: Any) -> Any:
        for i in range(self.retries):
            try:
                return fn(*a)
            except Exception:  # transient network / API errors
                if i == self.retries - 1:
                    raise
                time.sleep(2**i)

    def fetch(self, series_id: str, force: bool = False) -> pd.Series:
        csv = self.raw_dir / f"{series_id}.csv"
        if csv.exists() and not force:
            df = pd.read_csv(csv, index_col=0, parse_dates=True)
            return df.iloc[:, 0].rename(series_id)
        s = self._retry(self._fred.get_series, series_id).rename(series_id).dropna()
        info = self._retry(self._fred.get_series_info, series_id)
        s.index.name = "date"
        s.to_frame().to_csv(csv)
        self._write_meta(series_id, info, s)
        return s

    def _write_meta(self, sid: str, info: pd.Series, s: pd.Series) -> None:
        mp = self.raw_dir / "metadata.json"
        meta = json.loads(mp.read_text()) if mp.exists() else {}
        meta[sid] = {
            "title": str(info.get("title")),
            "units": str(info.get("units")),
            "frequency": str(info.get("frequency_short")),
            "seasonal_adjustment": str(info.get("seasonal_adjustment_short")),
            "vintage_date": date.today().isoformat(),
            "n_obs": int(len(s)),
            "first": str(s.index.min().date()),
            "last": str(s.index.max().date()),
        }
        mp.write_text(json.dumps(meta, indent=2))

    def fetch_many(self, ids: list[str], force: bool = False) -> dict[str, pd.Series]:
        return {i: self.fetch(i, force) for i in ids}
