"""Error metrics, all expressed in kW and in % of rated power.

nMAE (% of rated) is the wind-industry convention: it lets a 3.6 MW turbine be
compared with a 2 MW one. MAPE is not used because it explodes when power ~0.
"""
from __future__ import annotations

import numpy as np

from . import config as C


def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    return {
        "n": int(len(y_true)),
        "mae_kw": mae,
        "rmse_kw": rmse,
        "nmae_pct": 100 * mae / C.RATED_KW,
        "nrmse_pct": 100 * rmse / C.RATED_KW,
        # Bias matters most for anomaly detection: a biased reference curve
        # turns into a permanent false alarm (or a permanently masked fault).
        "bias_kw": float(np.mean(err)),
    }
