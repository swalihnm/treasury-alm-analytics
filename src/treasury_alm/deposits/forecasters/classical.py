"""Baselines, SARIMAX and LightGBM forecasters (monthly)."""

from __future__ import annotations

import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import ConvergenceWarning, ValueWarning
from statsmodels.tsa.statespace.sarimax import SARIMAX

from ..features import FEATURES, HORIZONS, targets
from .base import BaseForecaster


class NaiveForecaster(BaseForecaster):
    """Random-walk baselines for the (log) deposit level.

    mode='rw'    : zero growth (the classic random-walk forecast; MASE is defined against this).
    mode='drift' : trailing 12-month mean growth carried forward (random walk with drift).
    The series is seasonally adjusted, so a 'seasonal naive' would add noise, not information.
    """

    def __init__(self, mode: str = "rw") -> None:
        super().__init__()
        self.mode = mode
        self.name = f"naive_{mode}"

    def fit(self, df: pd.DataFrame, cutoff: pd.Timestamp) -> NaiveForecaster:
        return self

    def predict(self, df: pd.DataFrame, origins: pd.DatetimeIndex) -> pd.DataFrame:
        out = {}
        for h in HORIZONS:
            out[h] = 0.0 * df.loc[origins, "g"] if self.mode == "rw" else h * df.loc[origins, "g_m12"]
        return pd.DataFrame(out, index=origins)


class SarimaxForecaster(BaseForecaster):
    """SARIMAX(1,0,1) on monthly growth with exogenous policy/curve drivers (lagged one month).

    Orders are fixed (not searched) to keep the backtest honest and cheap; the series is
    seasonally adjusted so no seasonal terms. Multi-step forecasts hold exogenous drivers at their
    last observed value (no look-ahead); scenario work in `scenario.py` supplies explicit paths.
    """

    name = "sarimax"
    EXOG = ["dff3", "dff12", "slope"]

    def __init__(self) -> None:
        super().__init__()
        self._mu = self._sd = None
        self._params: np.ndarray | None = None

    def _exog(self, df: pd.DataFrame) -> pd.DataFrame:
        x = (df[self.EXOG] - self._mu) / self._sd
        return x

    def fit(self, df: pd.DataFrame, cutoff: pd.Timestamp) -> SarimaxForecaster:
        d = df.loc[:cutoff].dropna(subset=["g", *self.EXOG])
        self._mu, self._sd = d[self.EXOG].mean(), d[self.EXOG].std()
        x = self._exog(d).shift(1).dropna()
        y = d["g"].loc[x.index]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", (ConvergenceWarning, ValueWarning, UserWarning))
            res = SARIMAX(y, exog=x, order=(1, 0, 1), trend="c").fit(disp=False, maxiter=200)
        self._params = np.asarray(res.params)
        self._cutoff = cutoff
        return self

    def predict(self, df: pd.DataFrame, origins: pd.DatetimeIndex) -> pd.DataFrame:
        assert self._params is not None
        H = max(HORIZONS)
        rows = {}
        xall = self._exog(df).shift(1)
        for o in origins:
            hist = df.loc[:o].dropna(subset=["g", *self.EXOG])
            xh = xall.loc[hist.index].dropna()
            yh = hist["g"].loc[xh.index]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mod = SARIMAX(yh, exog=xh, order=(1, 0, 1), trend="c")
                res = mod.smooth(self._params)
                x_future = np.tile(self._exog(hist).iloc[[-1]].values, (H, 1))
                fc = res.get_forecast(steps=H, exog=x_future).predicted_mean.values
            cs = np.cumsum(fc)
            rows[o] = {h: cs[h - 1] for h in HORIZONS}
        return pd.DataFrame.from_dict(rows, orient="index")


class LightGBMForecaster(BaseForecaster):
    """Direct multi-horizon gradient boosting on lag/rolling features (small, heavily regularised).

    One model per horizon; training rows are origins t with t+h <= cutoff so no target leaks.
    ~500 monthly rows means shallow trees and strong regularisation; NaNs (spread before 2009)
    are handled natively by LightGBM.
    """

    name = "lightgbm"

    def __init__(self, seed: int = 42) -> None:
        super().__init__()
        self.seed = seed
        self.models: dict[int, lgb.LGBMRegressor] = {}

    def fit(self, df: pd.DataFrame, cutoff: pd.Timestamp) -> LightGBMForecaster:
        T = targets(df)
        for h in HORIZONS:
            lastorigin = cutoff - pd.DateOffset(months=h)
            idx = df.loc[:lastorigin].dropna(subset=["g_m12", "dff12"]).index
            y = T.loc[idx, h].dropna()
            m = lgb.LGBMRegressor(
                n_estimators=250,
                learning_rate=0.03,
                num_leaves=4,
                min_child_samples=15,
                subsample=0.8,
                subsample_freq=1,
                colsample_bytree=0.8,
                reg_lambda=5.0,
                random_state=self.seed,
                verbose=-1,
                n_jobs=1,
            )
            m.fit(df.loc[y.index, FEATURES], y)
            self.models[h] = m
        return self

    def predict(self, df: pd.DataFrame, origins: pd.DatetimeIndex) -> pd.DataFrame:
        X = df.loc[origins, FEATURES]
        return pd.DataFrame({h: self.models[h].predict(X) for h in HORIZONS}, index=origins)
