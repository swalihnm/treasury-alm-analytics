# Validation report: Treasury ALM analytics

**Prepared for:** an independent model reviewer · **Prepared by:** model developer (self-assessment; not an independent validation) · **Data vintage:** 2026-10-14 · **Scope:** curves (M2), deposit betas (M3), deposit forecasting (M4), anomaly detection (M5), NII drivers (M6), IRRBB engine (M7), NII-at-Risk (M8). Every number is in `results/summary.json`.

**Important context.** The bank in M7/M8 is **illustrative**: its balance sheet is an assumption (`config/balance_sheet.yaml`). Conclusions about “the bank” are demonstrations of the method, not findings about a real institution.

## 1. Overall assessment

| Model | Conceptual soundness | Data | Back-testing / outcome analysis | Overall rating |
|---|---|---|---|---|
| NSS curve | Sound | Adequate | Median fit error 2.9 bp | Satisfactory |
| Deposit betas | Sound, with sample-size limits | Adequate; regime break handled | Parameter recovery passes; cointegration evidence mixed | Satisfactory with findings |
| Deposit forecasters | Sound | Adequate | Beats naive; fails in 2022–23; drift RED | Satisfactory with findings |
| Anomaly detector | Sound | Adequate | 3/3 recall; low precision | Satisfactory with findings |
| NII drivers | Descriptive, not causal | Adequate (aggregate) | R² low; breaks significant | Use for explanation only |
| IRRBB engine | Sound; formulas verified against SRP31 | Illustrative bank | Closed-form and property tests pass | Satisfactory (method); results illustrative |
| NII-at-Risk | Sound with Gaussian caveat | Illustrative bank | Martingale and convergence checks pass | Satisfactory with findings |

## 2. Conceptual soundness

* **NSS** is fitted to implied par yields rather than treating par yields as spot rates (avoids long-end bias). Multi-start optimisation with bounds.
* **Betas.** Cycle betas are transparent but sensitive to boundary and lag choice; both are reported. The ECM is the right object for persistent rates, but its tests have low power at 62 observations (new regime). The Kalman specification on 6-month-average policy changes gives the cumulative beta under a uniform lag; a first specification (level regression with a random-walk intercept) was degenerate (variances collapsed to ~1e-11) and was rejected.
* **Forecasting.** Direct multi-horizon targets, leakage-safe features (test truncates data after each origin and asserts identical forecasts; test that `fit` ignores post-cutoff data). Ensemble weights come from 2012–15 only. Conformal intervals use only residuals already resolved at forecast time.
* **IRRBB.** Parameters were checked against the Basel Framework SRP31 (2026 text): shock table (SRP31.90), buckets (SRP31.96), core caps (SRP31.112), outlier test (SRP31.82), floor discretion (SRP31.93). The rotation-shock coefficients were *not* read directly (formulas are images on the website); they were recovered from the SRP31.92 worked example and are reproduced by a test. The floor level is my assumption.
* **Hull-White.** Drift fitted to today's curve; Monte Carlo reprices a 5-year discount factor within 0.1 bp. Calibration uses an exact-discretisation AR(1).

## 3. Data checks

28 FRED series (vintage 2026-10-14) validated with pandera (ranges, uniqueness, ordering); gap report: none. FDIC deposit-rate methodology break (2021-03/04) documented and kept as two regimes. QBP income verified as quarterly flow (Q4/Q1 ratio ≈ 1). NDR12MCD (“<100M”) is not level-comparable with CD12NRNJ. See `reports/data_quality.md`.

## 4. Back-testing and benchmarking

* **Deposit forecasts, 11 annual folds 2016–2026.** Mean MASE: ensemble 0.593, SARIMAX 0.643, LSTM 0.656, naive drift 0.740, LightGBM 0.776, random walk 1.036. The best model beats the random walk by 42.8% and drift by 19.9% (MASE-based).
* **Challengers that lose are retained:** the LSTM does not beat SARIMAX on average (wins only at 12 months); LightGBM is the weakest ML model; in the 2022–23 window the random walk (0.53) beats all models (drift 1.01, SARIMAX 1.02, LightGBM 1.11, LSTM 0.81, ensemble 0.83); in the 2020 window the ensemble and SARIMAX are best (1.14 vs random walk 2.35).
* **Intervals.** Ensemble coverage averaged over horizons: 80.6% (nominal 80%) and 94.2% (nominal 95%); the 12-month horizon under-covers at the 95% level for the LSTM (80%) and others (about 89%).
* **Anomalies.** Recall 3/3 for all weekly detectors; precision by episode 0.18–0.80; monthly Module 4 residuals miss March 2023.
* **Parameter recovery.** ECM simulation recovers β_LR (0.35) within 0.05 and α (−0.15) within 0.06 on three seeds; Kalman recovers a constant beta of 0.3 within 0.1.
* **Benchmarks for NII.** Vasicek vs Hull-White shows the value of fitting the curve (base NII falls 3.43 → 3.27bn under Vasicek mean reversion).

## 5. Sensitivity and stress analysis

IRRBB tornado (worst-case ΔEVE, pp of Tier 1): core maturity ±1 year 6.6, prepayment ×0.5/×1.5 6.1, core share ±10 pp 3.4, deposit beta ×0.5/×2 1.2, floor and deposit lag 0. NII (parallel up): deposit beta 2.9 pp, deposit lag 0.5, prepayment 0.3. Currency tables (US curve): worst ΔEVE −41.6% (USD), −46.0% (EUR, HKD), −54.4% (GBP), −62.2% (INR). Monte Carlo: pre-GFC calibration moves 95% EaR by ≤ 0.6 pp.

## 6. Findings and action plan

| # | Finding | Severity | Action |
|---|---|---|---|
| F1 | All six forecaster drivers have PSI > 0.25 vs the ≤ 2015 training sample (regime drift) | High | Retrain with post-2022 weighting; add regime indicator; review monthly |
| F2 | Models over-predict deposit growth in 2022–23; random walk wins in that window | Medium | Add rate-path-conditional drivers (balance-sheet-shrinkage/QT proxy); report window-specific MASE |
| F3 | Measured betas are national averages and low (5.7% savings, 24% CD in 2022–23 hike) | Medium | Obtain bank-specific rate data; sensitivity already shown |
| F4 | Cointegration evidence is mixed; new regime has 62 observations | Medium | Re-test as data accrue; treat long-run betas as indicative |
| F5 | Hypothesis “faster pass-through on cuts” not supported (old regime up-beta > down-beta) | Info | Report as is; zero-floor effect noted |
| F6 | Anomaly precision low; unverified single-week spikes; only three labelled events | Medium | Verify H.8 reclassification dates; extend event list |
| F7 | NII-margin regressions have low R² (8.9% levels, 1.2% differences) and break significantly | Medium | Use for explanation only; do not forecast with them |
| F8 | Illustrative bank is an outlier (41.6% of Tier 1) by construction | Info | Replace by actual balance sheet before any decision use |
| F9 | Hull-White Gaussian; residual kurtosis 7.4; weak deposit ARX (R² 0.07) | Medium | Add jump/stochastic-vol variant; improve deposit volume model |
| F10 | Basel rotation coefficients recovered from a worked example, floor level assumed | Low | Confirm coefficients with the published PDF formulas; confirm national floor |
| F11 | Non-USD results use the US curve | Low | Add local curves before using GBP/EUR/HKD/INR results |

## 7. Limitations of this report

Written by the developer; it is a self-assessment and a starting point for an independent review, not a substitute. No production data, no governance sign-off, no performance of the models out of sample beyond the backtest.
