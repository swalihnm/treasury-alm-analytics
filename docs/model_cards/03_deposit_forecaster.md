# Model card: deposit balance forecasters (M4)

**Owner:** A. Swalih NM · **Code:** `deposits/forecasters/`, `backtest.py` · **Status:** developed; awaiting review.

**Purpose.** Forecast US commercial-bank deposit growth at 1/3/6/12 months and compare statistical, machine-learning and deep-learning approaches fairly.

**Data.** DPSACBM027SBOG (monthly, SA, from 1976-07) with policy rate, curve slope and the savings-rate spread (2009 onwards). The LSTM uses weekly DPSACBW027SBOG with policy features lagged one month.

**Models.** Naive random walk and drift; SARIMAX(1,0,1) with exogenous policy-change and slope drivers (fixed orders); LightGBM (direct, shallow, regularised); TensorFlow LSTM (16 units, 52-week sequences, masked multi-horizon loss, early stopping, fixed seeds); ensemble with inverse-squared-MAE weights fitted on 2012–15 only.

**Protocol.** Rolling-origin expanding window, annual refits, 11 evaluation folds (2016–2026, including 2020 and 2022–23). MASE (scale = in-sample random-walk MAE, 2009–2015), sMAPE on levels, RMSE; rolling split-conformal intervals (36 resolved residuals).

**Performance (mean MASE over horizons).** Ensemble 0.593; SARIMAX 0.643; LSTM 0.656; naive drift 0.740; LightGBM 0.776; random walk 1.036. The ensemble improves on the random walk by 42.8% and on drift by 19.9% (27.8% at 12 months). Conformal coverage (mean over horizons): 80.6% at the 80% level and 94.2% at 95% for the ensemble.

**Honest negatives.** The LSTM does not beat SARIMAX on average (it wins at 12 months only); LightGBM is weakest at long horizons (small sample); during 2022–23 the random walk (MASE 0.53) beat every model, because the models extrapolate pre-2022 growth into a period of stagnating deposits; the 2020 inflow is unforecastable for all models.

**Limitations.** Direct multi-step models, not rate-path-conditional in the backtest (SARIMAX holds drivers at last values); sMAPE on levels is dominated by the level itself; the weekly versus monthly target definition differs slightly for the LSTM (4-week average growth).

**Monitoring.** Rolling 12-month MASE, PSI of drivers (currently RED: regime drift), interval coverage.
