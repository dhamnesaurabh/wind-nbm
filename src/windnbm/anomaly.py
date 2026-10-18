"""Underperformance detection on top of the normal-behaviour model.

Two tiers, because the two kinds of loss need different tools:
  Tier 1  STOPS - wind available, ~0 kW. A physical rule finds these with
          certainty (data.py); no ML needed.
  Tier 2  SUBTLE UNDERPERFORMANCE - the turbine runs but makes a few % less
          (blade soiling, pitch misalignment, yaw error, anemometer drift).
          Invisible point by point: residual noise is ~3-4% of rated power,
          the same size as the fault. This tier is what this module does.

Method (Tier 2): CUSUM on standardised NBM residuals.
CUSUM accumulates small, persistent negative deviations and ignores
zero-mean noise. It is the classical optimal test for a small, sustained
mean shift, which is exactly what degradation looks like.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from . import config as C
from .data import STATE_OPERATING

Z_CLIP = 4.0  # a single bad point can add at most 4 to the CUSUM -> one glitch never alarms alone


def eligible(df: pd.DataFrame) -> pd.Series:
    """Points the Tier-2 detector looks at: running turbine, usable wind."""
    return (df["state"] == STATE_OPERATING) & df["ws"].between(C.STOP_WS_MS, C.CUT_OUT_MS)


class ResidualStandardiser:
    """z = (residual - bin bias) / bin scale, per 1 m/s wind-speed bin.

    Per bin because residual noise is heteroscedastic: tiny near cut-in and
    at rated power, largest on the steep part of the curve (8-11 m/s), where
    a small wind error means a large power error. One global sigma would
    over-alarm mid-curve and be blind near rated.
    Fitted on VALIDATION normal data - residuals on training data are too
    optimistic (the model has seen those points).
    """

    def fit(self, ws: pd.Series, resid: pd.Series):
        b = np.floor(ws).astype(int)
        g = resid.groupby(b)
        self.bias_ = g.median()
        mad = g.apply(lambda s: np.median(np.abs(s - s.median())))
        self.scale_ = np.maximum(1.4826 * mad, 10.0)
        # fall back to global values for bins too sparse in validation
        self.g_bias_, self.g_scale_ = float(resid.median()), float(1.4826 * np.median(np.abs(resid - resid.median())))
        return self

    def transform(self, ws: pd.Series, resid: pd.Series) -> pd.Series:
        b = np.floor(ws.fillna(0)).astype(int)
        bias = b.map(self.bias_).fillna(self.g_bias_)
        scale = b.map(self.scale_).fillna(self.g_scale_)
        return (resid - bias) / scale


def cusum_lower(z: pd.Series, mask: pd.Series, k: float, h: float) -> tuple[pd.Series, pd.Series]:
    """One-sided (downward) CUSUM: S_t = max(0, S_{t-1} - z_t - k).

    Points outside `mask` (stops, low wind, gaps) are skipped and S is held,
    so a fault spanning a calm night is not forgotten. After an alarm S resets
    to 0, so one long fault produces repeated, countable alarm events.
    """
    zz = z.clip(-Z_CLIP, Z_CLIP).to_numpy()
    m = mask.to_numpy()
    s = np.zeros(len(zz))
    alarm = np.zeros(len(zz), dtype=bool)
    cur = 0.0
    for i in range(len(zz)):
        if m[i] and not np.isnan(zz[i]):
            cur = max(0.0, cur - zz[i] - k)
            if cur > h:
                alarm[i] = True
                cur = 0.0
        s[i] = cur
    return pd.Series(s, index=z.index), pd.Series(alarm, index=z.index)


def alarms_per_30d(alarm: pd.Series) -> float:
    days = (alarm.index.max() - alarm.index.min()).total_seconds() / 86400
    return float(alarm.sum() / days * 30)


def calibrate_h(z: pd.Series, mask: pd.Series, k: float, budget: float, grid=None) -> float:
    """Smallest threshold h whose alarm rate on known-healthy data is within budget.

    This turns an arbitrary-looking threshold into a business decision:
    "at most 1 false alarm per turbine per month", which operations can
    accept or change. Calibrated empirically (not from textbook ARL tables)
    because 10-minute residuals are autocorrelated, which breaks the i.i.d.
    assumption behind the tables.
    """
    grid = np.arange(2.0, 80.0, 0.5) if grid is None else grid
    for h in grid:
        _, a = cusum_lower(z, mask, k, h)
        if alarms_per_30d(a) <= budget:
            return float(h)
    return float(grid[-1])


def rolling_bias_adjust(z: pd.Series, mask: pd.Series, days: int = 14, lag_h: int = 24) -> pd.Series:
    """Ablation (REJECTED on validation): subtract a trailing 14-day median of z,
    lagged 1 day, to follow seasonal air-density drift. It also absorbs part of
    a real fault within days, which cost more detections than it saved."""
    med = z.where(mask).rolling(f"{days}D", min_periods=144 * 3).median().shift(lag_h * 6)
    return z - med.fillna(0)


def credited_detections(alarm_clean: pd.Series, alarms_injected: list, starts, window_h: float):
    """Detection is credited only if the injected run alarms inside the fault
    window AND before any alarm the clean data raises in that same window.
    Natural events therefore cannot inflate the score."""
    hits, delays = 0, []
    for s, a in zip(starts, alarms_injected):
        w = (a.index >= s) & (a.index < s + pd.Timedelta(hours=window_h))
        fi = a[w].idxmax() if a[w].any() else None
        fc = alarm_clean[w].idxmax() if alarm_clean[w].any() else None
        if fi is not None and (fc is None or fi < fc):
            hits += 1
            delays.append((fi - s).total_seconds() / 3600)
    return hits / max(len(starts), 1), delays


# ------------------------------------------------------------ comparator
class IsolationForestDetector:
    """Comparator: physics-free unsupervised detection on raw SCADA.

    Isolation Forest scores how easy a point is to isolate in (wind, power,
    turbulence) space. It needs no power-curve model, which is its appeal.
    To compare fairly it gets the SAME persistence logic (alarm when >= half
    of the last 6 h of eligible points are flagged) and the SAME false-alarm
    budget, calibrated on the same validation data.
    """
    features = ["ws", "power", "ti"]

    def __init__(self, window: int = 36, frac: float = 0.5):
        self.window, self.frac = window, frac

    def fit(self, df_train: pd.DataFrame):
        X = df_train.loc[eligible(df_train), self.features]
        self.model_ = IsolationForest(n_estimators=300, random_state=C.SEED).fit(X)
        return self

    def score(self, df: pd.DataFrame) -> pd.Series:
        m = eligible(df) & df[self.features].notna().all(axis=1)
        s = pd.Series(np.nan, index=df.index)
        s[m] = self.model_.score_samples(df.loc[m, self.features])
        return s

    def alarms(self, score: pd.Series, tau: float) -> pd.Series:
        flag = (score < tau).astype(float).where(score.notna())
        valid = flag.dropna()
        frac = valid.rolling(self.window, min_periods=self.window // 2).mean()
        raw = (frac >= self.frac)
        # alarm event = rising edge, then a refractory window of the same length
        alarm = pd.Series(False, index=score.index)
        last = None
        for t in raw[raw & ~raw.shift(1, fill_value=False)].index:
            if last is None or (t - last) >= pd.Timedelta(minutes=10 * self.window):
                alarm[t] = True
                last = t
        return alarm

    def calibrate(self, score_val: pd.Series, budget: float) -> float:
        for q in np.linspace(0.20, 0.0005, 120):
            tau = float(np.nanquantile(score_val, q))
            if alarms_per_30d(self.alarms(score_val, tau)) <= budget:
                self.tau_ = tau
                return tau
        self.tau_ = float(np.nanmin(score_val))
        return self.tau_


# ------------------------------------------------------------ fault injection
def inject_fault(df: pd.DataFrame, start, hours: float, kind: str, magnitude: float) -> pd.DataFrame:
    """Return a copy of df with a synthetic fault of known start and size.

    There are no labelled faults in this dataset, so we create ground truth.
      efficiency_loss : power x (1 - m) while running in partial load (< 12 m/s).
                        Mimics soiling / pitch offset. Above rated the pitch
                        system recovers the loss, so the physics says: no effect there.
      anemometer_drift: measured wind x (1 + m). Turbine is healthy, sensor lies.
                        Included on purpose - the NBM cannot tell this apart from
                        a real loss (an honest limitation, see write-up).
    """
    out = df.copy()
    end = pd.Timestamp(start) + pd.Timedelta(hours=hours)
    win = (out.index >= start) & (out.index < end)
    if kind == "efficiency_loss":
        m = win & (out["state"] == STATE_OPERATING) & (out["ws"] < C.PARTIAL_LOAD_MAX_MS)
        out.loc[m, "power"] = out.loc[m, "power"] * (1 - magnitude)
    elif kind == "anemometer_drift":
        out.loc[win, "ws"] = out.loc[win, "ws"] * (1 + magnitude)
    else:
        raise ValueError(kind)
    return out
