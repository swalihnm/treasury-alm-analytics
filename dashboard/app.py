"""Streamlit dashboard: run with `streamlit run dashboard/app.py`.

Pages read results/summary.json (written by the pipeline); the IRRBB page also re-runs the engine live
with interactive assumption sliders. The bank is ILLUSTRATIVE (assumptions, not data).
"""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from treasury_alm import FIGURES, RESULTS
from treasury_alm.irrbb.config import load_balance_sheet, load_basel
from treasury_alm.irrbb.outlier_test import IRRBBEngine, Overrides

st.set_page_config(page_title="Treasury ALM Analytics", layout="wide")
S = json.loads((RESULTS / "summary.json").read_text()) if (RESULTS / "summary.json").exists() else {}
page = st.sidebar.radio(
    "Page", ["Overview", "Deposit betas", "Forecasts", "Anomalies", "NII drivers", "IRRBB scenarios", "NII-at-Risk"]
)
st.sidebar.caption("Real data: FRED (rates, deposits, FDIC income). The bank balance sheet is illustrative.")


def img(name: str) -> None:
    p = FIGURES / name
    if p.exists():
        st.image(str(p))


if page == "Overview":
    st.title("Treasury ALM analytics")
    st.write(
        "Deposit behaviour, NII forecasting and Basel IRRBB stress testing on US data. Numbers come from `results/summary.json`."
    )
    if S:
        d = S["data"]
        st.metric("FRED series", d["n_series"])
        st.caption(
            f"Data vintage {d['vintage_date']}. Median NSS fit error {S['curves']['nss']['median_rmse_bp']:.1f} bp."
        )
    img("01_curves.png")
elif page == "Deposit betas":
    st.header("Deposit betas")
    rows = pd.DataFrame(S["betas"]["cycle_betas"])
    lag = st.select_slider("Lag adjustment (months)", [0, 3, 6], value=6)
    st.dataframe(
        rows[rows.lag_months == lag][["cycle", "product", "delta_policy_pp", "delta_deposit_pp", "beta"]],
        use_container_width=True,
    )
    img("03_cycle_betas.png")
    img("02_time_varying_beta.png")
    st.subheader("Asymmetry (up vs down beta)")
    st.dataframe(pd.DataFrame(S["betas"]["asymmetry"]).T)
elif page == "Forecasts":
    st.header("Deposit forecasting backtest")
    m = S["forecasting"]["metrics"]
    st.dataframe(
        pd.DataFrame({n: {f"MASE {h}m": v[h]["MASE"] for h in v} for n, v in m.items()}).T.style.format("{:.3f}")
    )
    st.caption(
        "Best model vs naive: "
        + json.dumps({k: v for k, v in S["forecasting"]["headline"].items() if k.startswith("mean_vs")})
    )
    img("04_forecast_mase.png")
    img("05_forecast_vs_actual.png")
    st.subheader("Interval coverage (nominal 80% / 95%)")
    st.json(S["forecasting"]["intervals"])
elif page == "Anomalies":
    st.header("Deposit-flow anomalies")
    img("06_anomalies.png")
    st.dataframe(
        pd.DataFrame(
            {k: {a: b for a, b in v.items() if not isinstance(b, dict)} for k, v in S["anomalies"]["detectors"].items()}
        ).T
    )
elif page == "NII drivers":
    st.header("NII drivers and attribution")
    img("07_nii_drivers.png")
    st.json(S["nii_drivers"]["attribution_2022_23_cycle"])
elif page == "IRRBB scenarios":
    st.header("IRRBB (Basel standardised) — illustrative bank")
    basel = load_basel()
    c1, c2, c3 = st.columns(3)
    cur = c1.selectbox(
        "Currency (shock table)", sorted(basel.shock_sizes), index=sorted(basel.shock_sizes).index("USD")
    )
    beta = c2.slider("Deposit beta multiplier", 0.0, 3.0, 1.0, 0.1)
    floor = c3.selectbox("Rate floor", ["maturity_dependent", "zero", "none"])
    core = c1.slider("Core share shift (pp)", -20, 20, 0) / 100
    mat = c2.slider("Core maturity shift (years)", -2.0, 2.0, 0.0, 0.5)
    cpr = c3.slider("Prepayment speed multiplier", 0.5, 1.5, 1.0, 0.1)
    r = IRRBBEngine(
        basel=basel,
        bs=load_balance_sheet(),
        ov=Overrides(
            beta_mult=beta,
            core_share_delta=core,
            core_maturity_delta=mat,
            cpr_scale=cpr,
            floor_mode=floor,
            currency=cur,
        ),
    ).run()
    t = pd.DataFrame(r["eve"]).T
    st.dataframe(t.style.format("{:.2f}"))
    ot = r["outlier_test"]
    (st.error if ot["is_outlier"] else st.success)(
        f"Outlier test: worst loss {ot['worst_loss_pct_tier1']:.1f}% of Tier 1 vs {ot['threshold_pct_tier1']:.0f}% threshold -> {'OUTLIER' if ot['is_outlier'] else 'pass'} ({r['worst_scenario']})"
    )
    st.dataframe(pd.DataFrame(r["nii"]).T.style.format("{:.3f}"))
    img("08_irrbb.png")
    img("09_repricing_gap.png")
else:
    st.header("NII-at-Risk (Hull-White Monte Carlo)")
    mc = S["monte_carlo"]
    st.dataframe(pd.DataFrame(mc["hull_white"]["ear"]).T[["base_bn", "p05_bn", "EaR95_pct_base", "EaR99_pct_base"]])
    st.json(mc["calibration"]["main"])
    img("10_monte_carlo.png")
