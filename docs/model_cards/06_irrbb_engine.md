# Model card: Basel IRRBB engine (M7)

**Owner:** A. Swalih NM · **Code:** `irrbb/` · **Status:** developed; awaiting independent review. **Applied to an ILLUSTRATIVE bank only.**

**Purpose.** Measure ΔEVE (six Basel scenarios) and ΔNII (two scenarios) and apply the 15%-of-Tier-1 outlier test.

**Method.** Product classes (cash, floating, fixed ladders, amortising loans with S-curve prepayment, NMDs with a core/non-core split and beta-linked rates) generate cash flows slotted into 19 buckets and 12-month interest paths. EVE = PV(assets) minus PV(liabilities) at the NSS zero curve (2026-09) plus per-product base-calibrated spreads. Shocks follow SRP31.90 to .93 with a configurable floor. The currency table selects shock sizes.

**Results (illustrative bank, USD).** Base EVE 6.0bn, base NII 3.48bn, Tier 1 9.0bn. ΔEVE (% of Tier 1): parallel up −41.6, steepener −36.4, short up −5.1, short down +4.6, flattener +24.5, parallel down +31.1. **Outlier test: fails** (worst loss 41.6% vs the 15% threshold). ΔNII: +10.0% (parallel up) and −12.1% (parallel down), so NII and EVE move in opposite directions. Most sensitive assumptions for worst ΔEVE: core maturity (swing 6.6 pp), prepayment speed (6.1 pp), core share (3.4 pp), deposit beta (1.2 pp); for NII: deposit beta (2.9 pp). Rate floors and deposit lag do not affect USD EVE.

**Verification.** Property tests (parallel shock constant, short shock tends to 0 at long tenors, floors respected); zero-coupon ΔEVE equals the closed-form value; SRP31.92 worked example reproduced; base EVE equals book value of rate-sensitive items; core-deposit caps enforced.

**Limitations.** The bank is an assumption, deliberately mortgage-heavy with low-beta deposits (not re-tuned after seeing results); CPR is constant within a scenario; no term-deposit redemption option and no commercial-margin treatment; non-USD results use the US curve; the Basel rotation formulas were recovered from a worked example, not read directly.
