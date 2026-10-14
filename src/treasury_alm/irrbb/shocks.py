"""Basel IRRBB standardised interest-rate shocks (Basel Framework SRP31.90-31.93).

Shapes (decay parameter x = 4 years, SRP31.91):
  parallel    : +/- R_par at every tenor
  short       : R_short * exp(-t/x)                 (largest at the short end, -> 0 at long tenors)
  long        : R_long  * (1 - exp(-t/x))           (largest at the long end; used only inside rotations)
  steepener   : -0.65*|R_short|*S_s(t) + 0.90*|R_long|*S_l(t)
  flattener   : +0.80*|R_short|*S_s(t) - 0.60*|R_long|*S_l(t)
with S_s(t) = exp(-t/x), S_l(t) = 1 - S_s(t). The rotation coefficients reproduce the SRP31.92 worked
example (100bp shocks at t = 3.5y: steepener +25.4bp, flattener -1.6bp); see tests/test_irrbb.py.

Post-shock floor (SRP31.93): national discretion, must be <= 0. Modes: none | zero | maturity_dependent.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArr = NDArray[np.float64]


@dataclass(frozen=True)
class FloorPolicy:
    mode: str = "maturity_dependent"  # none | zero | maturity_dependent
    immediate_pct: float = -1.0
    end_years: float = 20.0

    def floor(self, t: FloatArr) -> FloatArr:
        t = np.asarray(t, dtype=float)
        if self.mode == "none":
            return np.full_like(t, -np.inf)
        if self.mode == "zero":
            return np.zeros_like(t)
        if self.mode == "maturity_dependent":
            return self.immediate_pct * (1 - np.minimum(t / self.end_years, 1.0))
        raise ValueError(f"unknown floor mode {self.mode}")


class BaseShock(ABC):
    name: str

    @abstractmethod
    def bp(self, t: FloatArr) -> FloatArr:
        """Shock in basis points at tenors t (years)."""


def s_short(t: FloatArr, x: float = 4.0) -> FloatArr:
    return np.exp(-np.asarray(t, dtype=float) / x)


@dataclass(frozen=True)
class Parallel(BaseShock):
    size_bp: float
    sign: int = 1
    name: str = "parallel"

    def bp(self, t: FloatArr) -> FloatArr:
        return np.full(np.shape(t), self.sign * self.size_bp, dtype=float)


@dataclass(frozen=True)
class Short(BaseShock):
    size_bp: float
    sign: int = 1
    x: float = 4.0
    name: str = "short"

    def bp(self, t: FloatArr) -> FloatArr:
        return self.sign * self.size_bp * s_short(t, self.x)


@dataclass(frozen=True)
class Long(BaseShock):
    size_bp: float
    sign: int = 1
    x: float = 4.0
    name: str = "long"

    def bp(self, t: FloatArr) -> FloatArr:
        return self.sign * self.size_bp * (1 - s_short(t, self.x))


@dataclass(frozen=True)
class Rotation(BaseShock):
    short_bp: float
    long_bp: float
    c_short: float
    c_long: float
    x: float = 4.0
    name: str = "rotation"

    def bp(self, t: FloatArr) -> FloatArr:
        ss = s_short(t, self.x)
        return self.c_short * abs(self.short_bp) * ss + self.c_long * abs(self.long_bp) * (1 - ss)


def steepener(short_bp: float, long_bp: float, x: float = 4.0, c: tuple[float, float] = (-0.65, 0.90)) -> Rotation:
    return Rotation(short_bp, long_bp, c[0], c[1], x, "steepener")


def flattener(short_bp: float, long_bp: float, x: float = 4.0, c: tuple[float, float] = (0.80, -0.60)) -> Rotation:
    return Rotation(short_bp, long_bp, c[0], c[1], x, "flattener")


def standard_scenarios(sizes: dict[str, float], x: float = 4.0, coeffs: dict[str, tuple[float, float]] | None = None) -> dict[str, BaseShock]:
    """The six Basel EVE scenarios for a currency's shock sizes {parallel, short, long} (bp)."""
    cs = coeffs or {}
    st = steepener(sizes["short"], sizes["long"], x, cs.get("steepener", (-0.65, 0.90)))
    fl = flattener(sizes["short"], sizes["long"], x, cs.get("flattener", (0.80, -0.60)))
    return {
        "parallel_up": Parallel(sizes["parallel"], +1, "parallel_up"),
        "parallel_down": Parallel(sizes["parallel"], -1, "parallel_down"),
        "steepener": st,
        "flattener": fl,
        "short_up": Short(sizes["short"], +1, x, "short_up"),
        "short_down": Short(sizes["short"], -1, x, "short_down"),
    }


def apply_shock(zero_pct: FloatArr, tenors: FloatArr, shock: BaseShock, floor: FloorPolicy) -> FloatArr:
    """Shocked zero rates (percent): max(base + shock/100, floor(t)); floors never *raise* an unshocked rate above base."""
    shocked = zero_pct + shock.bp(tenors) / 100.0
    fl = floor.floor(tenors)
    # floor binds only on the shocked rate (an already-below-floor base rate is left unchanged)
    return np.where(shocked < fl, np.minimum(fl, np.maximum(zero_pct, fl)), shocked)
