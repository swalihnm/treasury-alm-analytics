# Assumptions register

Owner of every item: model developer (A. Swalih NM) until an independent reviewer is assigned. "Sensitivity" cites the
analysis that quantifies it (`results/summary.json` section in brackets). Items marked **ILLUSTRATIVE** are not data.

## A. Data

| ID | Assumption | Basis / source | Sensitivity | 
|---|---|---|---|
| D1 | QBP income series (QBPQYNTIY, QBPQYTIY) are quarterly flows, not year-to-date | Tested: mean Q4/Q1 NII ratio within year ≈ 1, not ≈ 4 [`data.qbp_flow_check`] | If wrong, all NII ratios would be ~4x off in Q4 |
| D2 | Old (2009–Mar 2021, weekly) and new (Apr 2021–, monthly) FDIC deposit-rate series are separate regimes, never spliced for betas | FDIC methodology change; documented in `reports/data_quality.md` | Betas are compared within regime only |
| D3 | The old/new **savings** series *are* spliced for one forecasting feature (`spread`), levels within 3 bp at the boundary | Policy rate ≈ 0 at the boundary | Feature used by LightGBM only |
| D4 | Weekly series are averaged to months; stamped on the first of the month | FRED convention | Minor |
| D5 | Average assets = mean of opening and closing balance; NII ratio = 4 × NII / average assets | QBP gives end-of-period assets | Level differs from USNIM, which uses earning assets (ours ≈ 0.84 × USNIM) |
| D6 | NDR12MCD (“12-month CD <100M”) is treated as the current 12M CD national rate | FRED title | Not comparable in level with CD12NRNJ |

## B. Curves

| ID | Assumption | Basis | Sensitivity |
|---|---|---|---|
| C1 | NSS is fitted to the *zero* curve; par yields are implied with semi-annual coupons; bills <1Y use simple compounding | Constant-maturity yields are par yields | Median fit error 2.9 bp |
| C2 | FRED GS yields are treated as par yields of Treasuries at constant maturity | FRED definition | – |
| C3 | Fit window starts 2001-07 (first month with GS1M) and needs ≥ 6 tenors | Data availability | – |

## C. Deposit behaviour (Module 3 / 4)

| ID | Assumption | Basis | Sensitivity |
|---|---|---|---|
| B1 | Rate cycles: zig-zag on FEDFUNDS, 0.75 pp turning threshold; boundaries tightened to within 0.15 pp of the extremes | Reproduces the spec's sanity dates | Cycle beta changes with boundary choice (lag 0/3/6 reported) |
| B2 | “Cumulative beta” uses 6-month lag adjustment when exported to IRRBB | Pass-through is gradual | Lag-0 betas are 26% (savings) and 31% (CD) higher than lag-6 for the 2022–23 hike |
| B3 | Measured betas are *national average posted rates*, which understate what rate-sensitive depositors receive | FDIC national rate averages all banks incl. those paying ≈ 0 | IRRBB tornado: deposit beta ×0.5 / ×2 |
| B4 | **ILLUSTRATIVE** retail-transactional beta = 0.25 × measured savings beta | Not observable in FRED | Immaterial in EVE (beta rows move worst-case ΔEVE by ≤ 1.2 pp) |
| B5 | **ILLUSTRATIVE** wholesale-NMD beta = 12M CD beta | Proxy | as above |
| B6 | Monthly deposit models train on 1976-07 onwards; evaluation 2016–2026, 2012–15 warm-up | Longer history covers several hiking cycles | See validation report: regime drift |
| B7 | Deposit growth in the Monte Carlo is an ARX(1) in lagged growth, 3-month policy change and slope (R² 0.07 on 1976–2026) with iid normal noise | Tractable path simulation | The rate effect is tiny in this specification; volume risk is dominated by noise |

## D. Basel IRRBB (Module 7)

| ID | Assumption | Basis | Sensitivity |
|---|---|---|---|
| I1 | 19 buckets, midpoints, six shock scenarios, USD/GBP/EUR/HKD/INR sizes, core-deposit caps, 15% Tier 1 outlier threshold | Verified against Basel Framework SRP31 (in force 2026-01-01) on 2026-10-14: SRP31.90, .96, .112, .82 | – |
| I2 | Rotation coefficients (steepener −0.65/0.90, flattener 0.80/−0.60) | Recovered from the SRP31.92 worked example (+25.4 bp / −1.6 bp at 3.5y); formulas are images in the web text, so not read directly | Unit test reproduces the example |
| I3 | Post-shock floor: −1.00% at t=0 rising linearly to 0% at 20 years (mode configurable: none / zero / maturity_dependent) | **My assumption**; SRP31.93 only requires floor ≤ 0 | No effect on USD results (rates stay well above floors) |
| I4 | EVE discounting at the NSS zero curve plus a per-product spread calibrated so base PV = book value | Removes arbitrary margin from the base | ΔEVE isolates rate effect |
| I5 | Core deposits run off linearly over 2×M years (average maturity M ≤ cap); non-core is overnight | Simple slotting consistent with the caps | Core maturity ±1y: ±6.6 pp of Tier 1 on worst ΔEVE |
| I6 | Core share: 80% / 60% / 40% (retail transactional / non-transactional / wholesale); core maturity 4 / 3 / 2 years | **ILLUSTRATIVE**, inside the caps | Core share ±10 pp: ±3.4 pp |
| I7 | Mortgage CPR: logistic S-curve in refinancing incentive (min 4%, max 40%, steepness 2/pp, midpoint 1 pp); constant over each scenario | **ILLUSTRATIVE** parameters | CPR ×0.5 / ×1.5: ±6.1 pp |
| I8 | NII: instantaneous parallel ±200 bp, constant balance sheet, 12 months; deposit rates adjust exponentially with 3-month time constant | SRP31 NII measure; lag is an assumption | Deposit lag 1m/6m: ≤ 0.5 pp of NII |
| I9 | Term-deposit early-withdrawal option ignored | Simplification | Understates optionality |
| I10 | Non-USD currencies use the same US zero curve with that currency's shock sizes | No non-USD curves in scope | Results for GBP/EUR/HKD/INR are a stress parameterisation, not a local-currency bank |
| I11 | **ILLUSTRATIVE BANK**: USD 100bn assets, Tier 1 USD 9bn, balances/rates in `config/balance_sheet.yaml`; chosen before results were seen and **not re-tuned** | Typical mid-sized US bank shape | The bank is a deliberate outlier (mortgage-heavy, low-beta deposits) |

## E. Monte Carlo (Module 8)

| ID | Assumption | Basis | Sensitivity |
|---|---|---|---|
| M1 | One-factor Hull-White; (a, σ) from AR(1) on 3-month T-bill rate, 1990-01 to 2026-09: a = 0.109, σ = 65 bp/√yr | Exact-discretisation OLS | Pre-GFC calibration (a = 0.195, σ = 69 bp) changes 95% EaR by ≤ 0.6 pp |
| M2 | Gaussian shocks; residuals are fat-tailed (excess kurtosis 7.4, Jarque-Bera p ≈ 0) | Standard HW | Tail risk understated |
| M3 | Base path = deterministic forward path; EaR measured against it | Convention | – |
| M4 | 10,000 antithetic paths, 36 monthly steps, seed 20260930 | Convergence plot | Year-1 95% EaR within ±0.05 pp from 2,000 paths up; 99% EaR within ±0.25 pp |
