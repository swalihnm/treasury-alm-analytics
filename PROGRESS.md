# Project progress log — treasury-alm-analytics

_Last updated: 2026-10-14. Single running record of what is done, key results, decisions and open items.
Every number below is copied from `results/summary.json` (written by code), not typed by hand._

## Status by phase

| Phase | Scope | Status |
|---|---|---|
| 0 | Scaffold, tooling, CI, Docker | Partial: package, pyproject, tests, git done. **CI, Makefile, Docker still to do** |
| 1 | M1 data platform | Done (16 FRED series, pandera validation, `reports/data_quality.md`) |
| 2 | M2 yield curve | Done (NSS median fit error 2.9 bp; PCA 3 factors = 99.4%) |
| 3 | M3 deposit betas | Done (cycles, ECM, asymmetry, Kalman beta, YAML export) |
| 4 | M4 forecasting + M5 anomalies | Done (5 models, 11 eval folds, conformal intervals, 4 detectors) |
| 5 | M6 NII drivers | Done (driver regressions, USNIM check, rate/volume attribution) |
| 6 | M7 IRRBB engine | **In progress** (Basel params verified, config + shocks written; products/EVE/NII next) |
| 7 | M8 Monte Carlo | Not started |
| 8 | Governance, API, dashboard, README | Not started |

## Key results so far (honest, including unfavourable ones)

**Data.** 16 series, vintage 2026-10-14. QBP income verified as quarterly flow (Q4/Q1 ratio ~1, not ~4).
The FDIC deposit-rate series (old weekly 2009–2021-03, new monthly 2021-04–) are kept separate.

**Curves.** NSS fitted to 303 months (2001-07 to 2026-09); median RMSE 2.9 bp, 95th pct 6.4 bp.
PCA variance shares: 88.1% / 10.1% / 1.3%.

**Deposit betas (national average posted rates, so low).**
- Savings cumulative beta: ~1.4–1.9% in 2015–18 hike, ~5.7–7.2% in 2022–23 hike. 12M CD: ~20% vs 24–31%.
- Cycle dates found algorithmically match the spec (hike 2015-11..2018-12; cut 2019-07..2020-04; hike 2022-03..2023-08; cut from 2024-08).
- **Hypothesis NOT supported:** cuts were *not* passed on faster than hikes (old regime up-beta > down-beta, p<0.02). Partly a zero-floor effect (savings rates near 0.1% cannot fall).
- ECM/cointegration evidence is mixed (Engle-Granger rejects cointegration in most samples; Johansen is more favourable). Reported as is.
- Beta rises with rate level in point estimates (e.g. new-regime savings 4.5% low-level vs 8.3% high-level) but the interaction is not statistically significant.
- Kalman time-varying beta: first specification was degenerate (variances collapsed); respecified on 6-month-average policy changes in bp, now well-behaved.

**Deposit forecasting (rolling-origin, 2016–2026, 11 folds).** Mean MASE: ensemble 0.593, SARIMAX 0.643,
LSTM 0.656, naive-drift 0.740, LightGBM 0.776, random walk 1.036. Ensemble beats drift by ~20% and random walk by ~43%.
- LSTM does **not** beat SARIMAX on average (it wins at the 12-month horizon only); LightGBM is the weakest ML model (small sample).
- In the 2022–23 hiking window the zero-growth random walk beat every model (deposits stagnated); the models over-predict growth there.
- Conformal coverage is close to nominal at short horizons, **under-covers at 12 months** (95% target → 89%).

**Anomalies.** All four detectors catch all 3 pre-labelled events (Lehman 2008, pandemic 2020, SVB 2023), recall 3/3.
Precision by episode is low at z>4 (0.27) because of isolated single-week small-bank spikes (cause unverified, possibly reclassification). Module-4 monthly residuals flag 2020 but **miss March 2023**.

**NII drivers.** Industry NII/assets: 2.33% (2021Q4) → 2.97% (2023Q4), peak 3.05% in 2022Q4.
Rate effect = ~99% of the change in NII over the cycle; volume ~1%. Industry funding-cost beta 41% (vs 6% for national savings rate).
Rate level and slope explain only ~9% of NII-margin variation in levels; break dummies (2008, 2020) lift R² to 0.35 and make rate/slope insignificant. USNIM correlation 0.74.

**Basel parameters.** Verified against SRP31 (2026 edition): shock table matches spec (USD 200/300/225), 19 buckets, core caps, 15% outlier threshold.
Rotation coefficients (−0.65/0.90, 0.80/−0.60) recovered from the SRP31.92 worked example. Floor level (−1% rising to 0 at 20y) is **my assumption**, not Basel text.

## Decisions made while the user was away
1. Monthly models train from 1976-07; evaluation 2016–2026 with 2012–15 warm-up (ensemble weights, conformal calibration).
2. Headline forecasting baseline = naive drift (stronger) as well as random walk; both reported.
3. Savings-rate spread feature uses a documented old/new splice (levels within 3 bp at the boundary); only for LightGBM.
4. Event windows for anomaly validation fixed in code before looking at detector output.
5. IRRBB discounting: per-product spread calibrated so base PV = book value; cash flows slotted into the 19 Basel buckets.

## Open items / known limitations
- Add Makefile, Dockerfile, docker-compose, GitHub Actions CI, pre-commit.
- ruff and `mypy --strict` not yet run on the package; coverage not yet measured.
- Remaining modules: M7 engine, M8 Monte Carlo, charts, model cards, assumptions register, validation report, FastAPI, Streamlit, README, final `summary.json` audit and resume bullets.
- API key lives only in gitignored `.env`; it was pasted in chat, so consider rotating it.
