"""Loading, regularising and labelling the SCADA data."""
from __future__ import annotations

import pandas as pd

from . import config as C

STATE_MISSING = "missing"
STATE_LOW_WIND = "low_wind"      # below cut-in: zero power is expected, not a fault
STATE_STOPPED = "stopped"        # wind available, no power: stop/curtailment
STATE_OPERATING = "operating"


def load_raw(path=C.RAW_CSV) -> pd.DataFrame:
    """Read the Kaggle CSV and give the columns short, unit-free names."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = ["ts", "power", "ws", "theo", "wd"]
    df["ts"] = pd.to_datetime(df["ts"], format="%d %m %Y %H:%M")
    return df.set_index("ts").sort_index()


def to_regular_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Re-index onto a complete 10-minute grid.

    Why: lag features ("power one hour ago") must mean one hour ago. On the raw
    file a shift(6) silently jumps across the 3.9% of missing timestamps.
    Missing rows become NaN and are labelled, never interpolated, because a
    filled value would look like a real measurement to every downstream model.
    """
    grid = pd.date_range(df.index.min(), df.index.max(), freq="10min")
    out = df.reindex(grid)
    out.index.name = "ts"
    return out


def clean_and_label(df: pd.DataFrame) -> pd.DataFrame:
    """Add an operating-state label from physical rules only.

    The rules use no model output, so the label cannot be circular with the
    anomaly detector that is evaluated later.
    """
    out = df.copy()
    # 57 rows have power down to -2.5 kW: the idle turbine drawing auxiliary
    # power from the grid. Physically real, but irrelevant for production.
    out["power"] = out["power"].clip(lower=0)

    state = pd.Series(STATE_OPERATING, index=out.index)
    state[out["ws"] < C.CUT_IN_MS] = STATE_LOW_WIND
    state[(out["ws"] >= C.STOP_WS_MS) & (out["power"] <= C.STOP_POWER_KW)] = STATE_STOPPED
    state[out["power"].isna() | out["ws"].isna()] = STATE_MISSING
    out["state"] = state
    return out


def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological train / validation / test split (no shuffling, ever)."""
    train = df[df.index < C.TRAIN_END]
    val = df[(df.index >= C.TRAIN_END) & (df.index < C.VAL_END)]
    test = df[df.index >= C.VAL_END]
    return train, val, test


def load_dataset() -> pd.DataFrame:
    """One call that every script uses, so all stages see identical data."""
    return clean_and_label(to_regular_grid(load_raw()))
