# Model card: NII drivers and rate/volume attribution (M6)

**Owner:** A. Swalih NM · **Code:** `pnl/` · **Status:** developed; awaiting review.

**Purpose.** Explain what drives banks' net interest margin and split changes in NII into volume and rate effects.

**Data.** FDIC Quarterly Banking Profile via FRED: NII, total interest income, total assets (all insured institutions, 1984-Q2 to 2026-Q2); interest expense = income minus NII; USNIM for validation to 2020-Q3.

**Method.** NII over average assets (annualised); OLS with HAC errors on policy rate, slope, lagged 4-quarter change, a rolling funding-cost-beta proxy (mechanically linked to NII, so reported separately) and break dummies; ADF/KPSS; Chow tests (2008-Q4, 2020-Q1) and CUSUM; rate/volume/mix attribution (exact reconstruction, max error 4e-11).

**Results.** Correlation with USNIM 0.74 in levels (our ratio is about 0.84 times USNIM because USNIM divides by earning assets). Policy rate and slope explain **8.9%** of NII-margin variation in levels (R² 0.089); with break dummies R² is 0.345 and rate and slope become insignificant; the 4-quarter-difference model has R² 0.012. All level series are non-stationary (ADF p 0.14 to 0.34); both breaks are significant (Chow p below 1e-7). Over 2021-Q4 to 2023-Q4 NII/assets rose from 2.33% to 2.97% (peak 3.05% in 2022-Q4); asset yields rose from 2.48% to 5.27% and funding cost from 0.14% to 2.30%; **99% of the change in NII was a rate effect and 0.8% volume**. The industry funding-cost beta against fed funds was 41%, versus 5.7% for the national savings rate (funding includes CDs and wholesale).

**Limitations.** Aggregate data only (so "mix" is the pure interaction term); level regressions are exposed to common trends (Durbin-Watson 0.06 to 0.23); margin is dominated by regime shifts rather than the rate level.
