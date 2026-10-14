"""`make report`: figures for the README / dashboard, drawn only from results/summary.json and processed data.

Palette: reference categorical slots (blue, orange, aqua, magenta, ...), light surface, secondary ink for text,
no dual axes, thin marks, direct labels where few series (see dataviz skill).
"""

from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from . import FIGURES, PROCESSED
from .results import load_summary

C = {
    "blue": "#2a78d6",
    "orange": "#eb6834",
    "aqua": "#1baf7a",
    "yellow": "#eda100",
    "magenta": "#e87ba4",
    "violet": "#6250d6",
    "red": "#e34948",
}
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK2,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": INK,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "legend.frameon": False,
    }
)


def _save(fig: Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / f"{name}.png", dpi=150)
    plt.close(fig)


def fig_curves(S: dict[str, Any]) -> None:
    p = pd.read_parquet(PROCESSED / "nss_params.parquet").set_index("date")
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].plot(p.index, p["rmse_bp"], color=C["blue"], lw=1.4)
    ax[0].axhline(S["curves"]["nss"]["median_rmse_bp"], color=C["orange"], lw=1, ls="--")
    ax[0].text(
        p.index[5],
        S["curves"]["nss"]["median_rmse_bp"] + 0.3,
        f"median {S['curves']['nss']['median_rmse_bp']:.1f} bp",
        color=INK2,
    )
    ax[0].set(title="NSS fit error by month (par yields)", ylabel="RMSE (bp)")
    v = np.array(S["curves"]["pca"]["variance_share"]) * 100
    ax[1].bar(["Level", "Slope", "Curvature"], v, color=C["blue"], width=0.55)
    for i, x in enumerate(v):
        ax[1].text(i, x + 1, f"{x:.1f}%", ha="center", color=INK2)
    ax[1].set(title="PCA of monthly yield changes", ylabel="variance explained (%)", ylim=(0, 100))
    _save(fig, "01_curves")


def fig_betas(S: dict[str, Any]) -> None:
    tv = pd.read_parquet(PROCESSED / "tv_beta.parquet").set_index("date")
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), sharey=False)
    cols = {"savings_old": C["blue"], "money_market_old": C["aqua"], "cd12_old": C["orange"]}
    for k, c in cols.items():
        s = tv[f"{k}|beta_smoothed"].dropna()
        ax[0].plot(s.index, s, color=c, lw=1.4, label=k.replace("_old", "").replace("_", " "))
    ax[0].set(title="Time-varying beta, old FDIC series (2009-2021)", ylabel="beta")
    ax[0].legend()
    for k, c in {"savings_new": C["blue"], "cd12_new": C["orange"]}.items():
        s = tv[f"{k}|beta_smoothed"].dropna()
        ax[1].plot(s.index, s, color=c, lw=1.4, label=k.replace("_new", "").replace("_", " ") + " (Kalman)")
        r = tv[f"{k}|rolling"].dropna()
        ax[1].plot(r.index, r, color=c, lw=1, ls=":", label=k.replace("_new", "").replace("_", " ") + " (rolling 36m)")
    ax[1].set(title="Time-varying beta, new FDIC series (2021-)", ylabel="beta")
    ax[1].legend(fontsize=8)
    _save(fig, "02_time_varying_beta")
    rows = pd.DataFrame(S["betas"]["cycle_betas"])
    rows = rows[rows.lag_months == 6]
    piv = rows.pivot_table(index="cycle", columns="product", values="beta")
    fig, ax = plt.subplots(figsize=(11, 3.8))
    piv = piv.reindex(sorted(piv.index, key=lambda s: s.split()[1]))
    order = ["savings_old", "money_market_old", "cd12_old", "savings_new", "cd12_new"]
    piv = piv[[c for c in order if c in piv.columns]]
    piv.plot.bar(
        ax=ax,
        color=[C["blue"], C["aqua"], C["orange"], C["violet"], C["magenta"]][: piv.shape[1]],
        width=0.8,
        edgecolor=SURFACE,
        linewidth=1.5,
    )
    ax.set(
        title="Cumulative deposit beta by rate cycle (6-month lag-adjusted)",
        ylabel="beta (share of policy move passed on)",
        xlabel="",
    )
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="", fontsize=8)
    _save(fig, "03_cycle_betas")


def fig_forecasts(S: dict[str, Any]) -> None:
    m = S["forecasting"]["metrics"]
    models = ["naive_rw", "naive_drift", "sarimax", "lightgbm", "lstm", "ensemble"]
    hs = ["1", "3", "6", "12"]
    colors = [MUTED, "#c3c2b7", C["blue"], C["orange"], C["aqua"], C["violet"]]
    fig, ax = plt.subplots(figsize=(11, 3.8))
    w = 0.13
    for i, n in enumerate(models):
        ax.bar(
            np.arange(4) + (i - 2.5) * w,
            [m[n][h]["MASE"] for h in hs],
            w,
            color=colors[i],
            label=n,
            edgecolor=SURFACE,
            linewidth=1.5,
        )
    ax.axhline(1, color=INK2, lw=0.8, ls="--")
    ax.set_xticks(range(4), [f"{h}m" for h in hs])
    ax.set(
        title="Rolling-origin backtest 2016-2026: MASE by horizon (lower is better; 1 = random-walk scale)",
        ylabel="MASE",
    )
    ax.legend(ncol=6, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    _save(fig, "04_forecast_mase")
    pr = pd.read_parquet(PROCESSED / "deposit_backtest_preds.parquet")
    tr = pd.read_parquet(PROCESSED / "deposit_truth.parquet").set_index("date")
    e = pr[pr.model == "ensemble"].set_index("origin")["3"]
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.plot(tr["3"].loc["2016":].index, tr["3"].loc["2016":], color=INK2, lw=1.2, label="actual 3-month growth")
    ax.plot(
        e.loc["2016":].index, e.loc["2016":], color=C["violet"], lw=1.4, label="ensemble forecast (made 3m earlier)"
    )
    for a, b, lab in [("2020-03-01", "2020-12-01", "2020 inflows"), ("2022-03-01", "2023-12-01", "2022-23 hikes")]:
        ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color=C["yellow"], alpha=0.12, lw=0)
        ax.text(pd.Timestamp(a), ax.get_ylim()[1] * 0.92, lab, color=INK2, fontsize=8)
    ax.set(title="US commercial-bank deposits: 3-month log growth, actual vs ensemble", ylabel="%")
    ax.legend(loc="lower left")
    _save(fig, "05_forecast_vs_actual")


def fig_anomalies(S: dict[str, Any]) -> None:
    f = pd.read_parquet(PROCESSED / "anomaly_flags.parquet").set_index("date").loc["2007":]
    fig, ax = plt.subplots(2, 1, figsize=(11, 5.2), sharex=True)
    for a, col, nm, c in [(ax[0], "g_large", "large banks", C["blue"]), (ax[1], "g_small", "small banks", C["orange"])]:
        a.plot(f.index, f[col], color=c, lw=0.8)
        fl = f[f["robust_z"]]
        a.scatter(fl.index, fl[col], s=26, color=C["red"], zorder=3, edgecolor=SURFACE, linewidth=1)
        a.set(ylabel=f"{nm}\nweekly growth (%)")
    for ev in S["anomalies"]["events"]:
        for a in ax:
            a.axvspan(pd.Timestamp(ev["start"]), pd.Timestamp(ev["end"]), color=C["yellow"], alpha=0.18, lw=0)
        ax[0].text(
            pd.Timestamp(ev["end"]) + pd.Timedelta(days=40),
            ax[0].get_ylim()[1] * 0.75,
            ev["name"].replace("_", " "),
            fontsize=8,
            color=INK2,
        )
    ax[0].set_title("Weekly deposit flows; red = robust-z flag (|z|>4), shaded = pre-labelled events")
    _save(fig, "06_anomalies")


def fig_nii(S: dict[str, Any]) -> None:
    q = pd.read_parquet(PROCESSED / "nii_quarterly.parquet").set_index("date")
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].plot(q.index, q["asset_yield"], color=C["blue"], lw=1.4, label="asset yield")
    ax[0].plot(q.index, q["funding_cost"], color=C["orange"], lw=1.4, label="funding cost")
    ax[0].plot(q.index, q["nii_ratio"], color=C["aqua"], lw=1.8, label="NII / avg assets")
    ax[0].plot(q.index, q["fedfunds"], color=MUTED, lw=1, ls="--", label="fed funds")
    ax[0].set(title="All FDIC-insured institutions (annualised % of assets)", ylabel="%")
    ax[0].legend(fontsize=8)
    y = pd.read_parquet(PROCESSED / "nii_attribution_yoy.parquet").set_index("date").loc["2019":]
    x = np.arange(len(y))
    ax[1].bar(x, y["volume"] / 1e3, color=C["blue"], label="volume", edgecolor=SURFACE, linewidth=1)
    ax[1].bar(
        x, y["rate"] / 1e3, bottom=y["volume"] / 1e3, color=C["orange"], label="rate", edgecolor=SURFACE, linewidth=1
    )
    ax[1].plot(x, y["d_nii"] / 1e3, color=INK, lw=1.2, label="total ΔNII (YoY)")
    ax[1].set_xticks(x[::4], [d.strftime("%Y-Q") + str((d.month - 1) // 3 + 1) for d in y.index[::4]], fontsize=8)
    ax[1].set(title="Rate / volume attribution of YoY ΔNII (mix ≈ 0, omitted)", ylabel="USD bn per quarter")
    ax[1].legend(fontsize=8)
    _save(fig, "07_nii_drivers")


def fig_irrbb(S: dict[str, Any]) -> None:
    r = S["irrbb"]
    names = list(r["eve"])
    v = [r["eve"][n]["pct_of_tier1"] for n in names]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.9))
    ax[0].bar(range(6), v, color=[C["red"] if x < 0 else C["blue"] for x in v], width=0.6)
    ax[0].axhline(-15, color=INK2, lw=0.9, ls="--")  # outlier test applies to losses only
    ax[0].text(-0.4, -13.5, "outlier threshold: loss > 15% of Tier 1", ha="left", va="bottom", color=INK2, fontsize=8)
    ax[0].set_xticks(range(6), [n.replace("_", "\n") for n in names], fontsize=8)
    ax[0].set(title="ΔEVE by Basel scenario (% of Tier 1)  - illustrative bank", ylabel="% of Tier 1")
    sens = pd.DataFrame(r["sensitivity"])
    g = sens.groupby("assumption")["eve_swing_vs_base_pp"].agg(["min", "max"]).sort_values("max")
    zero = g[(g["max"].abs() + g["min"].abs()) < 1e-9].index.tolist()
    g = g.drop(index=zero)
    g = g.loc[(g["max"].abs() + g["min"].abs()).sort_values().index]
    for i, (_k, row) in enumerate(g.iterrows()):
        ax[1].barh(i, row["max"] - row["min"], left=row["min"], color=C["blue"], height=0.55)
    ax[1].set_yticks(range(len(g)), g.index, fontsize=8)
    ax[1].axvline(0, color=INK2, lw=0.8)
    note = f"  (no effect on ΔEVE: {', '.join(zero)})" if zero else ""
    ax[1].set(title="Tornado: range of worst-case ΔEVE change (pp of Tier 1)", xlabel="pp vs base" + note)
    _save(fig, "08_irrbb")
    gap = pd.DataFrame(r["gap_table"])
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.bar(gap["bucket"], gap["gap"], color=[C["blue"] if x >= 0 else C["orange"] for x in gap["gap"]], width=0.7)
    ax.plot(gap["bucket"], gap["cumulative_gap"], color=INK, lw=1.2, marker="o", ms=3, label="cumulative gap")
    ax.set_xticks(gap["bucket"], [f"{m:g}" for m in gap["midpoint_years"]], fontsize=7, rotation=45)
    ax.set(title="Repricing gap by Basel bucket (assets - liabilities, USD bn; x = midpoint in years)", ylabel="USD bn")
    ax.legend()
    _save(fig, "09_repricing_gap")


def fig_mc(S: dict[str, Any]) -> None:
    mc = S["monte_carlo"]
    ann, base = np.load(PROCESSED / "mc_nii_annual.npy"), np.load(PROCESSED / "mc_nii_base.npy")
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.9))
    for i, (c, lab) in enumerate(zip([C["blue"], C["orange"], C["aqua"]], ["year 1", "year 2", "year 3"])):
        ax[0].hist(ann[:, i], bins=70, color=c, alpha=0.55, label=lab, edgecolor=SURFACE, linewidth=0.3)
        ax[0].axvline(base[0, i], color=c, lw=1.2, ls="--")
    ax[0].set(title="Simulated annual NII (dashed = base forward path)", xlabel="USD bn")
    ax[0].legend()
    cv = pd.DataFrame(mc["convergence"])
    ax[1].plot(cv["n_paths"], cv["EaR95_year1_pct"], color=C["blue"], marker="o", ms=4, label="EaR 95%, year 1")
    ax[1].plot(cv["n_paths"], cv["EaR99_year1_pct"], color=C["orange"], marker="o", ms=4, label="EaR 99%, year 1")
    ax[1].set(title="Convergence: EaR (% of base NII) vs number of paths", xlabel="paths", xscale="log", ylabel="%")
    ax[1].legend()
    _save(fig, "10_monte_carlo")


def run() -> None:
    S = load_summary()
    for f in (fig_curves, fig_betas, fig_forecasts, fig_anomalies, fig_nii, fig_irrbb, fig_mc):
        f(S)
