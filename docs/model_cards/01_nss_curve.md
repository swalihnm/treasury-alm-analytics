# Model card: Nelson-Siegel-Svensson yield curve (M2)

**Owner:** model developer (A. Swalih NM) · **Code:** `src/treasury_alm/curves/` · **Status:** developed, self-tested; awaiting independent review.

**Purpose.** Provide a smooth zero-coupon curve for discounting in the IRRBB engine and the Hull-White calibration, plus interpretable curve factors (level, slope, curvature).

**Data.** FRED constant-maturity Treasury yields GS1M, TB3MS, GS1–GS30, monthly; 303 monthly fits, 2001-07 to 2026-09.

**Method.** NSS zero-rate curve (β0..β3, τ1, τ2). Par yields are implied from the zero curve (semi-annual coupons; <1Y bills simple-compounded) and the par-yield error is minimised by bounded least squares (multi-start: OLS grid + previous month). PCA on monthly par-yield changes (3M–10Y, 1976-07 onwards).

**Performance.** Median fit RMSE 2.9 bp; 95th percentile 6.4 bp; max 11.7 bp. PCA: level 88.1%, slope 10.1%, curvature 1.3% (99.4% together). A par bond priced on the zero curve at its own par yield returns 100 to 1e-8 (unit test).

**Assumptions.** C1–C3 in the assumptions register.

**Limitations.** Fit quality is worst when few tenors are available and around 2008/2020 curve shapes; no instrument-level (coupon-stripping) curve; the Treasury curve is used as the risk-free curve for all currencies.

**Monitoring.** Monthly fit RMSE; flag if above 10 bp or if parameters hit bounds.
