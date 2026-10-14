import json

import pandas as pd
import pandera.errors
import polars as pl
import pytest

from treasury_alm.data import transforms as T
from treasury_alm.data.fred_client import FredClient, _load_key
from treasury_alm.data.validation import check_monotonic, detect_level_break, gap_report, positive_schema, rate_schema


class FakeFred:
    calls = 0

    def __init__(self, api_key):
        pass

    def get_series(self, sid):
        FakeFred.calls += 1
        return pd.Series(
            [1.0, 2.0, None, 4.0], index=pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01"])
        )

    def get_series_info(self, sid):
        return pd.Series({"title": "T", "units": "%", "frequency_short": "M", "seasonal_adjustment_short": "NSA"})


def test_fred_client_caches_and_writes_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr("treasury_alm.data.fred_client.Fred", FakeFred)
    c = FredClient(raw_dir=tmp_path, api_key="x")
    s = c.fetch("ABC")
    assert len(s) == 3 and FakeFred.calls == 1
    c.fetch("ABC")
    assert FakeFred.calls == 1  # served from cache
    meta = json.loads((tmp_path / "metadata.json").read_text())
    assert meta["ABC"]["n_obs"] == 3 and meta["ABC"]["vintage_date"]
    assert set(c.fetch_many(["ABC", "DEF"])) == {"ABC", "DEF"}


def test_fred_client_retries(tmp_path, monkeypatch):
    class Flaky(FakeFred):
        n = 0

        def get_series(self, sid):
            Flaky.n += 1
            if Flaky.n < 3:
                raise ConnectionError("boom")
            return super().get_series(sid)

    monkeypatch.setattr("treasury_alm.data.fred_client.Fred", Flaky)
    monkeypatch.setattr("time.sleep", lambda s: None)
    assert len(FredClient(raw_dir=tmp_path, api_key="x").fetch("Z")) == 3


def test_api_key_missing_raises(monkeypatch, tmp_path):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    monkeypatch.setattr("treasury_alm.data.fred_client.ROOT", tmp_path)
    with pytest.raises(RuntimeError):
        _load_key()
    (tmp_path / ".env").write_text("FRED_API_KEY=abc\n")
    assert _load_key() == "abc"


def test_transforms_weekly_to_monthly_and_wide(tmp_path):
    pd.DataFrame({"date": pd.date_range("2020-01-01", periods=8, freq="7D"), "S": range(8)}).to_csv(
        tmp_path / "S.csv", index=False
    )
    lf = T.read_raw("S", tmp_path)
    m = T.to_monthly(lf).collect()
    assert m["date"].to_list()[0].day == 1 and m["value"][0] == pytest.approx(2.0)
    q = T.to_quarterly(lf, "last").collect()
    assert q["value"][0] == 7
    assert "month_end" in T.month_end(lf).collect().columns
    w = T.wide({"a": lf, "b": T.to_monthly(lf)})
    assert set(w.columns) == {"date", "a", "b"}


def test_validation_schemas_and_gaps():
    df = pl.DataFrame(
        {
            "date": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 5, 1), "1mo", eager=True),
            "r": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    rate_schema(["r"]).validate(df)
    positive_schema(["r"]).validate(df)
    with pytest.raises(pandera.errors.SchemaError):
        rate_schema(["r"]).validate(df.with_columns(pl.col("r") * 100))
    assert check_monotonic(df)
    gappy = df.filter(pl.col("date") != pl.date(2020, 3, 1)).filter(pl.col("date") != pl.date(2020, 2, 1))
    assert gap_report(gappy, "r", 31) == [("2020-01-01", "2020-04-01")]
    assert detect_level_break(1.0, 3.0, 0.5) == 4.0
