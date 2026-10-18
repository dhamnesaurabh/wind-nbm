"""Figures for the write-up. One consistent, quiet style so the data speaks.

Palette: validated reference categorical slots 1-3 (blue, orange, aqua),
gray for context. Never more than three coloured series per chart.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "legend.frameon": False, "lines.linewidth": 1.6,
})


def _save(fig, name):
    C.FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(C.FIGURES / name)
    plt.close(fig)


def power_curve(df, curves: dict, name="f1_power_curve.png"):
    """Raw scatter + manufacturer curve + learned curves."""
    fig, ax = plt.subplots(figsize=(7, 4))
    op = df[df["state"] == "operating"]
    other = df[df["state"] == "stopped"]
    ax.scatter(op["ws"], op["power"], s=1, color=MUTED, alpha=0.15, rasterized=True, label="test records, Nov-Dec (operating)")
    ax.scatter(other["ws"], other["power"], s=2, color=INK2, alpha=0.4, rasterized=True, label="stopped (Tier-1 rule)")
    for (label, (x, y)), col in zip(curves.items(), [ORANGE, BLUE, AQUA]):
        ax.plot(x, y, color=col, lw=2, label=label)
    ax.set_xlim(0, 20); ax.set_ylim(-80, 3800)
    ax.set_xlabel("Wind speed (m/s)"); ax.set_ylabel("Active power (kW)")
    ax.set_title("The manufacturer curve over-promises; the learned curve follows reality")
    ax.legend(loc="lower right", markerscale=6, fontsize=8)
    _save(fig, name)


def seasonal_effect(monthly: pd.Series, name="f2_seasonal_density.png"):
    fig, ax = plt.subplots(figsize=(7, 3))
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    cols = [BLUE if m in (1, 2, 3, 11, 12) else (ORANGE if m in (6, 7, 8) else MUTED) for m in monthly.index]
    ax.bar([months[m - 1] for m in monthly.index], monthly.values, color=cols, width=0.7)
    for i, v in enumerate(monthly.values):
        ax.text(i, v + 10, f"{v:.0f}", ha="center", fontsize=7, color=INK2)
    ax.set_ylim(monthly.min() * 0.9, monthly.max() * 1.05)
    ax.set_ylabel("Median power at 7.5-8 m/s (kW)")
    ax.set_title("Same wind, ~15% more power in winter: denser cold air (no temperature in data)")
    ax.grid(axis="x", visible=False)
    _save(fig, name)


def error_by_ws(tab: pd.DataFrame, name="f3_nbm_error_by_ws.png"):
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for col, c in zip(tab.columns, [ORANGE, MUTED, BLUE]):
        ax.plot(tab.index, tab[col], color=c, marker="o", ms=3, label=col)
    ax.set_xlabel("Wind speed bin (m/s)"); ax.set_ylabel("MAE on test, normal operation (kW)")
    ax.set_title("Where the reference curve is wrong: the steep 7-12 m/s region")
    ax.legend()
    _save(fig, name)


def shap_bar(names, values, title, name):
    order = np.argsort(values)
    fig, ax = plt.subplots(figsize=(6, 0.35 * len(names) + 1))
    ax.barh(np.array(names)[order], np.array(values)[order], color=BLUE, height=0.6)
    for y, v in enumerate(np.array(values)[order]):
        ax.text(v, y, f"  {v:.0f} kW", va="center", fontsize=7, color=INK2)
    ax.set_xlabel("Mean |SHAP value| (kW of predicted power)")
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
    _save(fig, name)


def forecast_skill(tab: pd.DataFrame, name="f5_forecast_by_horizon.png"):
    """tab: index = horizon label, columns = model, values = nMAE %."""
    fig, ax = plt.subplots(figsize=(7, 3.4))
    x = np.arange(len(tab.index))
    for col, c in zip(["persistence", "ridge_ar", "lgbm"], [MUTED, ORANGE, BLUE]):
        ax.plot(x, tab[col], color=c, marker="o", ms=4, label=col)
        ax.text(x[-1] + 0.08, tab[col].iloc[-1], f"{tab[col].iloc[-1]:.1f}%", va="center", fontsize=7, color=INK2)
    ax.set_xticks(x, tab.index); ax.set_xlim(-0.2, len(x) - 0.5)
    ax.set_xlabel("Forecast horizon"); ax.set_ylabel("nMAE on test (% of rated)")
    ax.set_title("Without weather forecasts, ML only modestly beats persistence")
    ax.legend()
    _save(fig, name)


def backtest(tab: pd.DataFrame, name="f6_backtest_1h.png"):
    fig, ax = plt.subplots(figsize=(7, 3.2))
    x = np.arange(len(tab.index))
    for col, c in zip(["persistence", "lgbm"], [MUTED, BLUE]):
        ax.plot(x, tab[col], color=c, marker="o", ms=4, label=col)
    ax.set_xticks(x, tab.index)
    ax.set_ylabel("1-hour nMAE (% of rated)")
    ax.set_title("Expanding-window backtest at 1 hour: ML and persistence trade wins")
    ax.legend()
    _save(fig, name)


def detection_rates(tab: pd.DataFrame, name="f7_detection_rate.png"):
    """tab: index = scenario label, columns = detector, values = detection rate (0-1)."""
    fig, ax = plt.subplots(figsize=(7, 3.2))
    x = np.arange(len(tab.index)); w = 0.36
    for i, (col, c) in enumerate(zip(tab.columns, [BLUE, MUTED])):
        bars = ax.bar(x + (i - 0.5) * w, 100 * tab[col], width=w - 0.02, color=c, label=col)
        for b, v in zip(bars, tab[col]):
            ax.text(b.get_x() + b.get_width() / 2, 100 * v + 1.5, f"{100*v:.0f}%", ha="center", fontsize=7, color=INK2)
    labels = [s.replace(" loss ", " loss\n").replace(" drift ", " drift\n") for s in tab.index]
    ax.set_xticks(x, labels, fontsize=8); ax.set_ylim(0, 112)
    ax.set_ylabel("Faults detected within 7 days (%)")
    ax.set_title("Same false-alarm budget: physics residual + CUSUM vs Isolation Forest")
    ax.legend(loc="upper left"); ax.grid(axis="x", visible=False)
    _save(fig, name)


def event_panel(d: pd.DataFrame, cusum: pd.Series, h: float, alarms: pd.Series, title, name, fault_start=None):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.5, 4.2), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    a1.plot(d.index, d["expected"], color=ORANGE, lw=1.1, label="expected (healthy model)")
    a1.plot(d.index, d["power"], color=BLUE, lw=1.1, label="actual")
    a1.set_ylabel("Power (kW)"); a1.legend(loc="upper right", fontsize=7)
    a2.plot(cusum.index, cusum, color=INK2, lw=1.1)
    a2.axhline(h, color=ORANGE, lw=1, ls="--"); a2.text(cusum.index[0], h, " alarm threshold", va="bottom", fontsize=7, color=INK2)
    a2.set_ylabel("CUSUM")
    for t in alarms[alarms].index:
        for a in (a1, a2):
            a.axvline(t, color=INK, lw=0.8, alpha=0.6)
    if fault_start is not None:
        for a in (a1, a2):
            a.axvline(fault_start, color=AQUA, lw=1.5)
        a1.text(fault_start, a1.get_ylim()[1] * 0.92, " fault starts", fontsize=7, color=INK2)
    a1.set_title(title)
    fig.autofmt_xdate()
    _save(fig, name)
