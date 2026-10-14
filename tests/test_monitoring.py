import numpy as np
import pytest
from conftest import needs_data

from treasury_alm.monitoring import psi, status


def test_psi_zero_for_same_distribution_and_large_for_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(size=5000)
    assert psi(a, rng.normal(size=5000)) < 0.05
    assert psi(a, rng.normal(2.0, 1, 5000)) > 0.25


def test_status_bands():
    assert [status(v, 0.1, 0.25) for v in (0.05, 0.1, 0.2, 0.3)] == ["GREEN", "GREEN", "AMBER", "RED"]


@needs_data
def test_monitoring_run(sandbox):
    from treasury_alm import monitoring

    if not (sandbox / "processed" / "deposit_backtest_preds.parquet").exists():
        pytest.skip("needs backtest output")
    monitoring.PROCESSED = sandbox / "processed"
    out = monitoring.run()
    assert set(out["input_drift"]) >= {"slope", "dff3"} and out["forecast_error"]["status"] in {"GREEN", "AMBER", "RED"}
