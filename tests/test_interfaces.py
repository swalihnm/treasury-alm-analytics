import pytest
from conftest import needs_data
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from treasury_alm.api.main import app as api
from treasury_alm.cli import app as cli

pytestmark = needs_data
client = TestClient(api)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_irrbb_endpoint_default():
    r = client.post("/irrbb/run", json={})
    assert r.status_code == 200
    j = r.json()
    assert set(j["eve"]) == {"parallel_up", "parallel_down", "steepener", "flattener", "short_up", "short_down"}
    assert "disclaimer" in j and set(j["nii"]) == {"parallel_up", "parallel_down"}


def test_irrbb_endpoint_overrides_and_subset():
    r = client.post(
        "/irrbb/run",
        json={
            "currency": "GBP",
            "scenarios": ["steepener"],
            "tier1_capital": 20,
            "beta_multiplier": 2,
            "floor_mode": "zero",
            "balance_sheet": {"assets": {"cash": {"balance": 9.0}}},
        },
    )
    j = r.json()
    assert r.status_code == 200 and list(j["eve"]) == ["steepener"] and j["tier1_bn"] == 20


@pytest.mark.parametrize(
    "body",
    [
        {"currency": "JPY"},
        {"scenarios": ["nope"]},
        {"floor_mode": "bad"},
        {"balance_sheet": {"assets": {"mortgages": None}}},
    ],
)
def test_irrbb_endpoint_validation_errors(body):
    assert client.post("/irrbb/run", json=body).status_code == 422


def test_cli_help_and_irrbb(sandbox):
    runner = CliRunner()
    assert "backtest" in runner.invoke(cli, ["--help"]).output
    res = runner.invoke(cli, ["irrbb"])
    assert res.exit_code == 0 and "parallel_up" in res.output
