# Model card: deposit-flow anomaly and regime detection (M5)

**Owner:** A. Swalih NM · **Code:** `deposits/anomaly.py` · **Status:** developed; awaiting review.

**Purpose.** Flag abnormal weekly deposit flows (large vs small banks) the way liquidity monitoring would.

**Method.** Robust one-sided z-score of detrended weekly growth (|z| > 4); Isolation Forest (fitted on 1985–2007 so all labelled events are out of sample); CUSUM; PELT changepoints (offline); robust z on the Module 4 monthly one-step errors.

**Validation.** Three events labelled *before* looking at detector output: Lehman flight-to-safety (2008-09-15 to 11-30), pandemic inflows (2020-03-11 to 06-30), regional bank stress (2023-03-08 to 04-19). Evaluation window 2008–2026.

**Results.** Every weekly detector catches all 3 events (recall 3/3), first flags within 0–1 weeks of the window start (PELT, being offline, flags 1–2 weeks before). Episode precision is low: robust-z 0.27 (3 of 11 episodes; 0.41 false-alarm weeks per calm year), CUSUM 0.21, Isolation Forest 0.18 (1.1 per year), PELT 0.80 (5 episodes). The extra episodes are mostly isolated single-week small-bank spikes whose cause I did not verify (possibly large/small reclassification). Threshold sensitivity: precision rises from 0.20 (z above 3) to 0.60 (z above 6) with recall unchanged at 3/3. **The Module 4 monthly residual layer flags 2020-03 and 2020-04 but misses March 2023** (large-bank inflows offset small-bank outflows in the aggregate).

**Limitations.** Three labelled events make precision and recall coarse; precision treats unlabelled genuine anomalies as false alarms; thresholds are conventional, not optimised.
