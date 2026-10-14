"""Vasicek benchmark:  dr = a (b - r) dt + sigma dW  (constant long-run mean b, no curve fit).

Used only as a simpler benchmark for Hull-White: it ignores today's curve (rates drift toward b), so
its NII distribution is centred differently, which is exactly why Hull-White is preferred for EaR.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Arr = NDArray[np.float64]


@dataclass
class Vasicek:
    a: float
    b: float  # long-run mean (decimal)
    sigma: float
    r0: float  # decimal

    def simulate(self, n_paths: int, n_months: int, seed: int, antithetic: bool = True) -> Arr:
        dt = 1 / 12
        rng = np.random.default_rng(seed)
        half = n_paths // 2 if antithetic else n_paths
        z = rng.standard_normal((half, n_months))
        if antithetic:
            z = np.stack([z, -z], axis=1).reshape(-1, n_months)  # interleaved: any prefix stays antithetic-balanced
        decay = np.exp(-self.a * dt)
        sd = self.sigma * np.sqrt((1 - np.exp(-2 * self.a * dt)) / (2 * self.a))
        r = np.full((z.shape[0], n_months + 1), self.r0)
        for k in range(n_months):
            r[:, k + 1] = self.b + (r[:, k] - self.b) * decay + sd * z[:, k]
        return r

    def zero_rates(self, r: Arr, tenor: float) -> Arr:
        """Closed-form Vasicek zero rate (percent) at fixed time-to-maturity."""
        a, b, s = self.a, self.b, self.sigma
        B = (1 - np.exp(-a * tenor)) / a
        lnA = (B - tenor) * (a**2 * b - s**2 / 2) / a**2 - s**2 * B**2 / (4 * a)
        return -(lnA - B * r) / tenor * 100.0
