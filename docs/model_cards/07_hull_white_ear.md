# Model card: Hull-White NII-at-Risk (M8)

**Owner:** A. Swalih NM · **Code:** `simulation/` · **Status:** developed; awaiting review. **Illustrative bank.**

**Purpose.** Replace six fixed scenarios by a distribution of 36-month NII outcomes.

**Method.** One-factor Hull-White with drift fitted to the NSS curve (Monte Carlo martingale test: 5-year discount factor within 0.1 bp). (a, σ) by AR(1) on the 3-month T-bill (1990–2026): a = 0.109 (half-life 6.4 years), σ = 65 bp per √year. 10,000 antithetic paths, 36 months. Dynamic balance sheet: deposits follow an ARX deposit-growth model (R² 0.07) around the base plan, surplus or deficit moves cash then wholesale funding; deposit rates follow measured betas with a 3-month lag; mortgages prepay on the S-curve using the path's 10Y rate. A Vasicek benchmark and a pre-GFC calibration are run for comparison.

**Results.** Base NII (forward path) is 3.53bn, 3.64bn and 3.69bn in years 1 to 3. 95% Earnings-at-Risk: 3.3%, 5.9% and 8.1% of base (0.12, 0.22 and 0.30bn); 99%: 4.7%, 8.4% and 11.4%; cumulative 3-year 5.4% (95%) and 7.7% (99%). Year-1 95% EaR is stable from 2,000 paths (3.27 to 3.29%), bootstrap SE 0.001bn. Pre-GFC calibration: 95% EaR 3.3, 5.8 and 7.6%. Vasicek (no curve fit) drifts toward its 1.8% long-run mean, giving a lower base NII and larger EaR (3.7, 8.3 and 13.0%), which is why Hull-White is preferred. Run time is under 2 seconds.

**Limitations.** Gaussian shocks against fat-tailed history (excess kurtosis 7.4); single factor (no independent slope risk); the ARX deposit model is weak, so deposit-volume risk is mostly noise; EaR is relative to the forward path, not to the Basel static scenarios (which show NII of +10% and −12% for ±200 bp instantaneous parallel shocks, much larger than the simulated 95% outcomes).
