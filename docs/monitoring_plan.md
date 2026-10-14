# Monitoring plan

Run monthly after new FRED data: `alm data --force && alm monitor` (results in `results/summary.json → monitoring`).
Thresholds are policy choices (assumptions), set in `src/treasury_alm/monitoring.py`.

| Check | Metric | AMBER | RED | Current status (2026-10) |
|---|---|---|---|---|
| Forecast error | Rolling 12-month MASE of the ensemble, 3-month horizon | > 0.80 (≈ naive-drift level) | > 1.00 (worse than random walk) | 0.37 → GREEN |
| Input drift | PSI of each driver, training (≤ 2015) vs last 36 months | 0.10 | 0.25 | All six drivers RED (PSI 2.1–5.9) |
| Beta stability | Relative change of filtered Kalman beta vs 36 months earlier | > 50% | > 100% | Current-regime series GREEN; the retired 2009–2021 series are not monitored |
| Interval calibration | Empirical coverage of 80% / 95% conformal intervals over last 36 months | outside ±5 pp | outside ±10 pp | Backtest: 80.6% / 94.2% (ensemble, mean over horizons) |
| Event watch | Robust-z / CUSUM flags on weekly flows | any flag → analyst review | flag in both large and small banks | See anomaly detector |

**Recalibration triggers:** any RED, or AMBER on two consecutive monthly runs. Beta re-estimation is also triggered by a
new policy-rate cycle (zig-zag identifies a new turning point). Basel parameter tables are re-verified against SRP31 at
each BCBS publication.

**Reading the current drift result.** The deposit forecasters were trained on a sample dominated by low-rate years; the
last 36 months are in a different rate regime, so every driver has drifted far from the training distribution. Forecast
error is currently low, but the drift flag says the model's coefficients have been learned on rates it rarely saw, so
extra caution is warranted when rates move sharply.

**Escalation:** AMBER → developer review within 5 working days; RED → model risk notified, results annotated as
“under review” until recalibrated and re-validated.
