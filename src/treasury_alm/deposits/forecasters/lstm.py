"""TensorFlow/Keras LSTM forecaster on weekly data (more observations than the monthly series).

Inputs  : 52 weeks of [weekly log growth, lagged policy rate, lagged 3-month policy change].
Outputs : 4 heads = growth of the 4-week-average deposit level over 4/13/26/52 weeks ahead
          (~1/3/6/12 months), the weekly analogue of the monthly target.
Policy-rate features use the *previous* month's average so a week never sees a month that has
not finished (no leakage). Loss is masked MSE so recent origins with unresolved long horizons
still train the short heads. Seeds are fixed and op-determinism enabled.
"""

from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import pandas as pd
import polars as pl
import tensorflow as tf
from numpy.typing import NDArray

from ... import PROCESSED
from ..features import HORIZONS
from .base import BaseForecaster

WEEKS = {1: 4, 3: 13, 6: 26, 12: 52}
SEQ = 52


def weekly_frame() -> pd.DataFrame:
    w = pl.read_parquet(PROCESSED / "weekly.parquet").to_pandas().set_index("date")["dep_all"].dropna()
    m = pl.read_parquet(PROCESSED / "monthly.parquet").to_pandas().set_index("date")["fedfunds"].dropna()
    ff_prev = m.shift(1)  # value known at the end of the previous month
    d3_prev = m.diff(3).shift(1)
    df = pd.DataFrame({"D": w})
    df["A"] = df["D"].rolling(4).mean()
    df["wg"] = 100 * np.log(df["D"]).diff()
    month = df.index.to_period("M").to_timestamp()
    df["ff"] = ff_prev.reindex(month).values
    df["dff3"] = d3_prev.reindex(month).values
    return df.dropna()


class LSTMForecaster(BaseForecaster):
    name = "lstm"

    def __init__(self, seed: int = 7, units: int = 16, epochs: int = 60) -> None:
        super().__init__()
        self.seed, self.units, self.epochs = seed, units, epochs
        self.wf = weekly_frame()
        self.model: tf.keras.Model | None = None
        self.history: dict[str, list[float]] = {}

    # ---- helpers -------------------------------------------------------------------------
    def _x(self) -> NDArray[np.float32]:
        return np.asarray(self.wf[["wg", "ff", "dff3"]].values, dtype=np.float32)

    def _y(self) -> tuple[np.ndarray, np.ndarray]:
        lA = np.log(self.wf["A"].values)
        ys, ms = [], []
        for h in HORIZONS:
            k = WEEKS[h]
            y = np.full(len(lA), np.nan)
            y[: len(lA) - k] = 100 * (lA[k:] - lA[:-k])
            ys.append(y)
            ms.append(~np.isnan(y))
        return np.column_stack(ys), np.column_stack(ms)

    def _seqs(self, xs: np.ndarray, idx: np.ndarray) -> np.ndarray:
        return np.stack([xs[i - SEQ + 1 : i + 1] for i in idx])

    def _build(self) -> tf.keras.Model:
        tf.keras.utils.set_random_seed(self.seed)
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:  # pragma: no cover
            pass
        inp = tf.keras.Input((SEQ, 3))
        h = tf.keras.layers.LSTM(self.units)(inp)
        h = tf.keras.layers.Dropout(0.2)(h)
        out = tf.keras.layers.Dense(len(HORIZONS))(h)
        m = tf.keras.Model(inp, out)

        def masked_mse(yt: tf.Tensor, yp: tf.Tensor) -> tf.Tensor:
            y, mask = yt[:, : len(HORIZONS)], yt[:, len(HORIZONS) :]
            return tf.reduce_sum(mask * tf.square(y - yp)) / (tf.reduce_sum(mask) + 1e-8)

        m.compile(tf.keras.optimizers.Adam(1e-3), loss=masked_mse)
        return m

    # ---- API -----------------------------------------------------------------------------
    def fit(self, df: pd.DataFrame, cutoff: pd.Timestamp) -> LSTMForecaster:
        cut_week = self.wf.index[self.wf.index <= (cutoff + pd.offsets.MonthEnd(0))][-1]
        n_cut = int(self.wf.index.get_loc(cut_week)) + 1
        xs_all = self._x()
        self._mu, self._sd = xs_all[:n_cut].mean(0), xs_all[:n_cut].std(0) + 1e-6
        xs = (xs_all - self._mu) / self._sd
        Y, M = self._y()
        # targets must resolve by the cutoff week
        for j, h in enumerate(HORIZONS):
            M[max(0, n_cut - WEEKS[h]) :, j] = False
        Y = np.where(M, Y, 0.0)
        self._ysd = np.array([Y[M[:, j], j].std() + 1e-6 for j in range(len(HORIZONS))])
        idx = np.arange(SEQ - 1, n_cut)
        idx = idx[M[idx].any(axis=1)]
        X = self._seqs(xs, idx).astype("float32")
        yt = np.hstack([Y[idx] / self._ysd, M[idx].astype(float)]).astype("float32")
        nval = max(int(0.15 * len(idx)), 50)
        self.model = self._build()
        cb = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
        h = self.model.fit(
            X[:-nval],
            yt[:-nval],
            validation_data=(X[-nval:], yt[-nval:]),
            epochs=self.epochs,
            batch_size=64,
            shuffle=False,
            callbacks=[cb],
            verbose=0,
        )
        self.history = {k: [float(v) for v in vs] for k, vs in h.history.items()}
        return self

    def predict(self, df: pd.DataFrame, origins: pd.DatetimeIndex) -> pd.DataFrame:
        assert self.model is not None
        xs = (self._x() - self._mu) / self._sd
        idx = []
        for o in origins:
            end = o + pd.offsets.MonthEnd(0)
            pos = int(self.wf.index.searchsorted(end, side="right")) - 1
            idx.append(pos)
        X = self._seqs(xs, np.array(idx)).astype("float32")
        p = self.model.predict(X, verbose=0) * self._ysd
        return pd.DataFrame(p, index=origins, columns=list(HORIZONS))
