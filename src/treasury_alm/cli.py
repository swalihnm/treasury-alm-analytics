"""`alm` command line: one command per Makefile target."""

from __future__ import annotations

import typer

app = typer.Typer(help="Treasury ALM analytics pipeline", no_args_is_help=True)


@app.command()
def data(force: bool = typer.Option(False, help="Re-download from FRED")) -> None:
    """Fetch FRED data, validate, write data/processed/*.parquet and reports/data_quality.md."""
    from .data.pipeline import run

    typer.echo(run(force))


@app.command()
def curves() -> None:
    """Fit NSS curves and PCA (M2)."""
    from .curves.pca import run

    typer.echo(run()["nss"])


@app.command()
def betas() -> None:
    """Deposit betas, ECM, Kalman beta; export config/derived/deposit_betas.yaml (M3)."""
    from .deposits.run_betas import run

    run()
    typer.echo("betas done")


@app.command()
def backtest(track: bool = typer.Option(True, help="Log to MLflow")) -> None:
    """Rolling-origin deposit forecasting backtest + conformal intervals (M4); then anomalies (M5)."""
    from .deposits import anomaly
    from .deposits.backtest import run

    typer.echo(run(track)["headline"])
    anomaly.run()


@app.command()
def train(track: bool = typer.Option(True)) -> None:
    """Alias of `backtest` (models are refit inside each fold)."""
    backtest(track)


@app.command()
def nii() -> None:
    """NII drivers and rate/volume attribution (M6)."""
    from .pnl.nii_drivers import run

    typer.echo(run()["attribution_2022_23_cycle"])


@app.command()
def irrbb() -> None:
    """Basel IRRBB engine: six EVE scenarios, two NII scenarios, outlier test, sensitivity (M7)."""
    from .irrbb.outlier_test import run

    r = run()
    typer.echo({k: round(v["pct_of_tier1"], 2) for k, v in r["eve"].items()})


@app.command()
def montecarlo(paths: int = 10_000) -> None:
    """Hull-White NII-at-Risk (M8)."""
    from .simulation.nii_at_risk import run

    typer.echo(run(paths)["hull_white"]["ear"]["year1"])


@app.command()
def monitor() -> None:
    """Forecast-error, PSI drift and beta-stability monitoring status."""
    from .monitoring import run

    out = run()
    typer.echo(
        {
            "forecast_error": out["forecast_error"]["status"],
            "drift": {k: v["status"] for k, v in out["input_drift"].items()},
        }
    )


@app.command()
def report() -> None:
    """Draw figures into reports/figures."""
    from .report import run

    run()


@app.command()
def quality() -> None:
    """Run the test suite with coverage and record counts in results/summary.json (engineering section)."""
    import json
    import subprocess
    import sys
    import xml.etree.ElementTree as ET

    from . import ROOT
    from .results import update_summary

    cov, xml = ROOT / "cov.json", ROOT / "junit.xml"
    rc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-W",
            "ignore",
            "--cov=treasury_alm",
            f"--cov-report=json:{cov}",
            f"--junitxml={xml}",
        ],
        cwd=ROOT,
    ).returncode
    root = ET.parse(xml).getroot()
    suite = root if root.tag == "testsuite" else root[0]
    tot = json.loads(cov.read_text())["totals"]
    out = {
        "tests_collected": int(suite.get("tests", 0)),
        "failures": int(suite.get("failures", 0)) + int(suite.get("errors", 0)),
        "skipped": int(suite.get("skipped", 0)),
        "coverage_pct": float(tot["percent_covered"]),
        "pytest_exit_code": rc,
        "ruff_exit_code": subprocess.run([sys.executable, "-m", "ruff", "check", "src", "tests"], cwd=ROOT).returncode,
        "mypy_strict_exit_code": subprocess.run(
            [sys.executable, "-m", "mypy", "src/treasury_alm"], cwd=ROOT
        ).returncode,
    }
    cov.unlink()
    xml.unlink()
    update_summary("engineering", out)
    typer.echo(out)


@app.command()
def serve(port: int = 8000) -> None:
    """Start the FastAPI service."""
    import uvicorn

    uvicorn.run("treasury_alm.api.main:app", host="0.0.0.0", port=port)


if __name__ == "__main__":
    app()
