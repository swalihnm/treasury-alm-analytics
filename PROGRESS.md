# Project progress log — treasury-alm-analytics

_Last updated: 2026-10-14. Single running record of what is done, what was verified, decisions taken, and open items.
Numbers are copied from `results/summary.json` (written by code), not typed from memory._

## Status by phase

| Phase | Scope | Status |
|---|---|---|
| 0 | Scaffold, tooling, CI, Docker | **Done** (pyproject, Makefile, CI workflow, Dockerfile, compose, pre-commit). Docker image **not built/verified** (daemon not running on this machine); CI workflow not yet run on GitHub |
| 1 | M1 data platform | Done: 28 FRED series, pandera validation, `reports/data_quality.md` |
| 2 | M2 yield curve | Done: NSS median fit error 2.9 bp; PCA 3 factors = 99.4% |
| 3 | M3 deposit betas | Done: cycles, ECM, asymmetry, level dependence, Kalman beta, YAML export |
| 4 | M4 forecasting + M5 anomalies | Done: 5 models + ensemble, 11 evaluation folds, conformal intervals, 4 detectors |
| 5 | M6 NII drivers | Done: regressions, USNIM check, breaks, rate/volume attribution |
| 6 | M7 IRRBB engine | Done: Basel parameters verified vs SRP31, 6 EVE + 2 NII scenarios, outlier test, tornado |
| 7 | M8 Monte Carlo | Done: Hull-White + Vasicek, 10,000 antithetic paths, convergence, martingale check |
| 8 | Governance, API, dashboard, README | Done: 7 model cards, assumptions register, validation report, monitoring plan + code, FastAPI, Streamlit, README |

## Verification performed

* `python -m treasury_alm.cli quality` → 66 tests passed, 0 failed, **97.1% coverage**, ruff exit 0, `mypy --strict` exit 0 (recorded in `summary.json → engineering`).
* Streamlit: all 7 pages rendered through Streamlit's headless test harness with no exceptions.
* FastAPI: live server started, `/health` and `POST /irrbb/run` answered correctly; 4 invalid-input cases return 422.
* Basel parameters checked against the BIS SRP31 web text (2026 edition). Rotation-shock coefficients recovered from the SRP31.92 worked example (formulas are images on the site).

## Key results (including unfavourable ones)

* **Betas:** savings 1.9% (2015–18 hike) vs 5.7% (2022–23 hike); CD 20.1% → 24.0% (6-month lag). Up-beta > down-beta in old regime: *the "faster on cuts" hypothesis is not supported.*
* **Forecasting:** ensemble MASE 0.593 vs random walk 1.036 (−42.8%) and drift 0.740 (−19.9%). LSTM (0.656) does not beat SARIMAX (0.643). In 2022–23 the random walk beat every model.
* **Monitoring:** PSI RED for all six drivers (regime drift vs ≤ 2015 sample); rolling forecast error GREEN (0.37).
* **Anomalies:** 3/3 events detected by every weekly detector; low precision (0.18–0.80). Module 4 residuals miss March 2023.
* **NII:** rate level + slope explain 8.9% of margin variation in levels; 99% of 2022–23 NII rise is a rate effect.
* **IRRBB (illustrative bank):** worst ΔEVE −41.6% of Tier 1 (parallel up) → outlier; ΔNII +10.0% / −12.1%.
* **NII-at-Risk:** 95% EaR 3.3% / 5.9% / 8.1% (years 1–3); 99% 4.7% / 8.4% / 11.4%.

## Decisions taken while the user was away

1. Monthly models train from 1976-07; evaluation 2016–2026; 2012–15 warm-up for ensemble weights and conformal calibration.
2. Both random-walk and drift baselines reported; headline improvement quoted against both.
3. Savings-rate spread feature uses a documented old/new splice (within 3 bp at the boundary), LightGBM only.
4. Anomaly event windows fixed in code before looking at detector output.
5. Illustrative bank fixed before results were seen and **not re-tuned** even though it is a large outlier.
6. IRRBB spreads calibrated so base PV = book; floor level (−1% → 0 at 20y) is my assumption.
7. `.env` holds the FRED key (gitignored); the key never appears in code or git history.

## Mistakes caught and fixed along the way

Zig-zag cycle detector mishandled the first/last leg (found by a unit test); first Kalman-beta spec was degenerate; a hard-coded "ruff/mypy clean" string in the quality record was replaced by real exit codes; API returned 500 on malformed balance-sheet overrides (now 422); chart label overlaps and misleading tornado colours fixed after viewing the renders.

## Open items / not verified

* Docker image build and docker-compose not tested (Docker daemon not running locally).
* GitHub Actions workflow not run (needs a `FRED_API_KEY` repo secret to build data for the integration tests).
* No GitHub remote configured; work is committed locally only.
* The FRED API key was pasted into chat; consider rotating it at fred.stlouisfed.org.
* Independent validation has not happened; `docs/validation_report.md` is a developer self-assessment.
