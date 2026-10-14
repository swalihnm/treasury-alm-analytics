"""Time-varying deposit beta via a state-space (Kalman filter) regression.

Model (in basis points):   D d_t = b_t * x_t + e_t,       e_t ~ N(0, s_e^2)
                           b_t   = b_{t-1} + v_t,         v_t ~ N(0, s_b^2)
where x_t = mean of the previous six monthly policy-rate changes. Because each policy change
enters six consecutive x_t with weight 1/6, b_t is the *cumulative* pass-through (beta) under a
uniform six-month lag. (A level regression with a free random-walk intercept was tried first
and is degenerate: the intercept absorbs all variation and the variances collapse to zero.)

The smoothed (two-sided) path is for history; the filtered (one-sided) path is what was knowable
in real time. Variances are estimated by maximum likelihood. Compare with
`betas.rolling_beta`, a window estimator, for robustness.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from statsmodels.tools.sm_exceptions import ConvergenceWarning, ValueWarning
from statsmodels.tsa.statespace.mlemodel import MLEModel

LAG_WINDOW = 6


class TVBeta(MLEModel):  # type: ignore[misc]
    def __init__(self, endog: pd.Series, exog: pd.Series) -> None:
        super().__init__(endog, k_states=1, initialization="approximate_diffuse")
        self["design"] = np.zeros((1, 1, self.nobs))
        self["design", 0, 0, :] = np.asarray(exog, dtype=float)
        self["transition"] = np.eye(1)
        self["selection"] = np.eye(1)
        self._v = float(np.var(np.asarray(endog, dtype=float)))

    @property
    def start_params(self) -> NDArray[np.float64]:
        return np.array([max(self._v, 1e-3), 1e-4])

    @property
    def param_names(self) -> list[str]:
        return ["var_obs", "var_beta"]

    def transform_params(self, unconstrained: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray(unconstrained) ** 2

    def untransform_params(self, constrained: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.sqrt(np.asarray(constrained))

    def update(self, params: NDArray[np.float64], **kwargs: Any) -> None:
        params = super().update(params, **kwargs)
        self["obs_cov", 0, 0] = params[0]
        self["state_cov", 0, 0] = params[1]


def regressors(dep: pd.Series, ff: pd.Series) -> pd.DataFrame:
    df_ = pd.concat([dep, ff], axis=1, keys=["d", "f"]).dropna()
    y = df_["d"].diff() * 100.0
    x = df_["f"].diff().shift(1).rolling(LAG_WINDOW).mean() * 100.0
    return pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()


def fit_tv_beta(dep: pd.Series, ff: pd.Series) -> dict[str, Any]:
    """Fit the TVP regression; returns smoothed/filtered beta paths and estimated variances."""
    z = regressors(dep, ff)
    z.index = pd.DatetimeIndex(z.index, freq="MS")
    mod = TVBeta(z["y"], z["x"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", (ConvergenceWarning, ValueWarning))
        res = mod.fit(disp=False, maxiter=500, method="nm")
        res = mod.fit(res.params, disp=False, maxiter=200)
    burn = LAG_WINDOW
    smooth = pd.Series(res.smoothed_state[0], z.index, name="beta_smoothed")
    filt = pd.Series(res.filtered_state[0], z.index, name="beta_filtered")
    se = pd.Series(np.sqrt(res.smoothed_state_cov[0, 0]), z.index, name="se")
    return {
        "smoothed": smooth.iloc[burn:],
        "filtered": filt.iloc[burn:],
        "smoothed_se": se.iloc[burn:],
        "params": dict(zip(mod.param_names, [float(p) for p in res.params])),
        "loglike": float(res.llf),
        "converged": bool(res.mle_retvals.get("converged", True)) if res.mle_retvals else True,
    }
