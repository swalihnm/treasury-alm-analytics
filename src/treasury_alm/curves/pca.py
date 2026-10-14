"""PCA of monthly yield-curve changes and the curve-factor pipeline (`make` stage: curves)."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from sklearn.decomposition import PCA

from .. import PROCESSED
from ..results import update_summary
from .nss import TENORS, fit_nss
from .zero_curve import ZeroCurve

PCA_COLS = ["tb3ms", "gs1", "gs2", "gs3", "gs5", "gs7", "gs10"]


def yield_change_pca(monthly: pd.DataFrame, start: str = "1976-07-01") -> dict[str, Any]:
    """PCA on first differences of par yields (bp). Returns variance shares and loadings."""
    d = monthly[PCA_COLS].loc[start:].dropna().diff().dropna() * 100.0
    pca = PCA(n_components=3).fit(d.values)
    load = pca.components_
    if load[0].sum() < 0:  # sign convention: level factor loads positively
        load[0] *= -1
    if load[1][-1] < load[1][0]:  # slope: long minus short positive
        load[1] *= -1
    return {
        "n_obs": int(len(d)), "start": str(d.index.min().date()), "tenors": PCA_COLS,
        "variance_share": pca.explained_variance_ratio_.tolist(),
        "cumulative_3": float(pca.explained_variance_ratio_.sum()),
        "loadings": {n: load[i].tolist() for i, n in enumerate(["level", "slope", "curvature"])},
    }


def run() -> dict[str, Any]:
    monthly = pl.read_parquet(PROCESSED / "monthly.parquet").to_pandas().set_index("date")
    monthly.index = pd.DatetimeIndex(monthly.index)
    rows, rmses, prev = [], [], None
    for dt, r in monthly.iterrows():
        cols = [c for c in TENORS if pd.notna(r.get(c))]
        if len(cols) < 6 or dt < pd.Timestamp("2001-07-01"):  # GS1M available from 2001-07
            continue
        mats = np.array([TENORS[c] for c in cols])
        y = np.array([r[c] for c in cols], dtype=float)
        p, rmse = fit_nss(mats, y, start=prev)
        prev = p.vector()
        rows.append({"date": dt, **{k: v for k, v in zip(["b0", "b1", "b2", "b3", "tau1", "tau2"], p.vector())},
                     "rmse_bp": rmse, "n_tenors": len(cols)})
        rmses.append(rmse)
    nss = pd.DataFrame(rows).set_index("date")
    nss.reset_index().pipe(pl.from_pandas).write_parquet(PROCESSED / "nss_params.parquet")
    feats = pd.DataFrame({"level": monthly["gs10"], "slope": monthly["gs10"] - monthly["tb3ms"],
                          "curvature": 2 * monthly["gs2"] - monthly["tb3ms"] - monthly["gs10"]}).dropna()
    feats.reset_index().pipe(pl.from_pandas).write_parquet(PROCESSED / "curve_features.parquet")
    pca = yield_change_pca(monthly)
    out = {
        "nss": {"n_months_fitted": len(nss), "first": str(nss.index.min().date()), "last": str(nss.index.max().date()),
                "median_rmse_bp": float(np.median(rmses)), "p95_rmse_bp": float(np.percentile(rmses, 95)),
                "max_rmse_bp": float(np.max(rmses))},
        "pca": pca,
    }
    last = nss.iloc[-1]
    from .nss import NSSParams
    zc = ZeroCurve.from_nss(NSSParams(*last[["b0", "b1", "b2", "b3", "tau1", "tau2"]].tolist()))
    out["latest_zero_curve"] = {"date": str(nss.index[-1].date()), "bucket_zero_pct": zc.zero(np.array(
        [0.25, 1, 2, 5, 10, 20, 30])).tolist(), "tenors": [0.25, 1, 2, 5, 10, 20, 30]}
    update_summary("curves", out)
    return out
