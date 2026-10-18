"""Feature engineering.

All rolling windows look backwards only (past and present), so no feature can
peek at the future.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Features for the normal-behaviour model (NBM): conditions that a healthy
# turbine's output depends on. Deliberately NO lagged power: if the model can
# see recent power, it learns to copy a fault and the residual disappears.
NBM_FEATURES = [
    "ws",          # the physics: P ~ 0.5 * rho * A * Cp * v^3
    "wd_sin", "wd_cos",   # direction -> terrain / wake effects
    "doy_sin", "doy_cos", # season -> air density proxy (no temperature in data)
    "hour_sin", "hour_cos",  # day/night -> atmospheric stability, wind shear
    "ti",          # turbulence proxy -> gusty wind gives less power than steady wind
]


def add_cyclic(df: pd.DataFrame) -> pd.DataFrame:
    """Encode angles and calendar as sin/cos so 359 deg sits next to 1 deg
    and 31 Dec sits next to 1 Jan."""
    out = df.copy()
    wd = np.deg2rad(out["wd"])
    out["wd_sin"], out["wd_cos"] = np.sin(wd), np.cos(wd)
    doy = 2 * np.pi * out.index.dayofyear / 365.25
    out["doy_sin"], out["doy_cos"] = np.sin(doy), np.cos(doy)
    hour = 2 * np.pi * (out.index.hour + out.index.minute / 60) / 24
    out["hour_sin"], out["hour_cos"] = np.sin(hour), np.cos(hour)
    return out


def add_turbulence(df: pd.DataFrame, window: int = 6) -> pd.DataFrame:
    """Turbulence-intensity proxy = std / mean of wind speed over the last hour.

    True TI needs the 1 Hz signal inside each 10-min record, which the dataset
    does not have; the spread of consecutive 10-min means is the best proxy.
    """
    out = df.copy()
    roll = out["ws"].rolling(window, min_periods=4)
    out["ti"] = (roll.std() / roll.mean().clip(lower=1.0)).fillna(0.0)
    return out


def build_nbm_frame(df: pd.DataFrame) -> pd.DataFrame:
    return add_turbulence(add_cyclic(df))
