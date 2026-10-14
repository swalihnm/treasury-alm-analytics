# Model card: deposit betas (M3)

**Owner:** A. Swalih NM · **Code:** `deposits/betas.py, ecm.py, kalman_beta.py, run_betas.py` · **Status:** developed; awaiting review.

**Purpose.** Quantify how much of a policy-rate move is passed to deposit rates, by product and rate cycle; feeds the IRRBB engine (`config/derived/deposit_betas.yaml`).

**Data.** FEDFUNDS monthly; FDIC national rates: old regime (savings, money market, 12M CD; weekly averaged to monthly, 2009-05 to 2021-03) and new regime (savings, 12M CD; monthly, 2021-04 onwards). Regimes are never spliced.

**Method.** (1) Zig-zag cycle identification; (2) cumulative cycle beta with 0/3/6-month lag adjustment; (3) ECM in one-step form, long-run β = −δ/α with delta-method HAC errors, Engle-Granger and Johansen tests; (4) distributed-lag up/down asymmetry with a HAC Wald test; (5) level dependence (split sample and interaction); (6) Kalman-filter time-varying beta on 6-month-average policy changes, plus a 36-month rolling benchmark; (7) parameter-recovery test on simulated ECM data.

**Results.** Savings cumulative beta (6-month lag): 1.9% (2015–18 hike), 5.7% (2022–23 hike), 1.7% and 1.9% in the cuts. 12M CD: 20.1% → 24.0% in the hikes. Up-beta exceeded down-beta in the old regime (savings 4.6% vs 1.4%, p = 0.016; CD 36.8% vs 13.3%, p = 0.012); no significant asymmetry in the new regime (p = 0.13 savings, 0.78 CD). **The hypothesis of faster pass-through on cuts than hikes is not supported.** Beta point estimates are higher at policy rates of 1% or more (e.g. new-regime savings 8.3% vs 4.5%) but the interaction term is insignificant (p 0.22–0.78). ECM long-run betas: new-regime savings 8.1%, 12M CD 41.9% (SE 26 pp); cointegration evidence is mixed (Engle-Granger p between 0.54 and 0.99 in every sample; Johansen rejects no-cointegration in 4 of 8 samples).

**Assumptions.** B1–B5 in the assumptions register.

**Limitations.** National *average* rates understate betas of rate-sensitive depositors; the new regime has only 62 monthly observations and one full cycle; zero-lower-bound floors bias down-betas; Engle-Granger tests have low power at this sample size; the first Kalman specification (level regression with random-walk intercept) was degenerate and was replaced.

**Monitoring.** Beta-stability check and new-cycle trigger (see monitoring plan).
