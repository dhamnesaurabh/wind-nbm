"""Short-term power forecasting (10 min to 6 h ahead).

Strategy: DIRECT multi-horizon - one model per horizon, trained to predict
power(t + h) from information available at time t. Chosen over a recursive
one-step model because recursive forecasts feed their own errors back in, and
those errors compound exactly at the longer horizons we care about.

Candidates per horizon:
  * Persistence       P(t+h) = P(t). The bar every wind forecast must clear;
                      at 10 min it is famously hard to beat.
  * Ridge AR(X)       Linear autoregression on the same features. Tests whether
                      the problem needs non-linearity at all.
  * LightGBM          Non-linear; can learn that a ramp near rated power
                      behaves differently from one near cut-in (power curve).
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C
from .data import STATE_STOPPED

POWER_LAGS = [1, 2, 3, 6, 12, 36, 144]   # 10 min .. 24 h back
WS_LAGS = [1, 2, 3, 6]


def build_forecast_features(df: pd.DataFrame, nbm_expected: pd.Series) -> pd.DataFrame:
    """Features known at time t. df must be on the regular 10-minute grid."""
    f = pd.DataFrame(index=df.index)
    p, ws = df["power"], df["ws"]
    f["p_0"] = p
    for k in POWER_LAGS:
        f[f"p_lag{k}"] = p.shift(k)
    f["ws_0"] = ws
    for k in WS_LAGS:
        f[f"ws_lag{k}"] = ws.shift(k)
    f["p_mean_1h"] = p.rolling(6, min_periods=4).mean()
    f["p_mean_3h"] = p.rolling(18, min_periods=12).mean()
    f["p_std_1h"] = p.rolling(6, min_periods=4).std()
    f["ws_mean_1h"] = ws.rolling(6, min_periods=4).mean()
    f["ws_diff_10m"] = ws.diff(1)
    f["ws_diff_1h"] = ws.diff(6)
    # Physics feature: what a healthy turbine would make at the current wind.
    # Lets the model separate "low wind" from "turbine stopped".
    f["nbm_expected_0"] = nbm_expected
    f["stopped_0"] = (df["state"] == STATE_STOPPED).astype(float)
    for c in ["wd_sin", "wd_cos", "hour_sin", "hour_cos", "doy_sin", "doy_cos"]:
        f[c] = df[c]
    return f


def make_target(df: pd.DataFrame, h: int) -> pd.Series:
    return df["power"].shift(-h)


def complete_rows(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """Rows usable by ALL models (no NaN in features or target).

    Ridge cannot take NaN; LightGBM can. Evaluating every model on the same
    rows keeps the comparison fair; the cost is ~5% of rows next to data gaps.
    """
    return X.notna().all(axis=1) & y.notna()


class Persistence:
    name = "persistence"

    def fit(self, X, y, *a, **k):
        return self

    def predict(self, X):
        return X["p_0"].to_numpy()


class RidgeAR:
    """Linear AR(X). Like LightGBM, it predicts the CHANGE P(t+h) - P(t)."""
    name = "ridge_ar"

    def fit(self, X, y, *a, **k):
        self.model_ = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 12)))
        self.model_.fit(X, y - X["p_0"])
        return self

    def predict(self, X):
        return np.clip(X["p_0"].to_numpy() + self.model_.predict(X), 0, C.RATED_KW)


class LightGBMForecaster:
    """Predicts the CHANGE in power, then adds it to the current power.

    Why the change and not the level: power 1 h ahead is ~92% correlated with
    power now (lag-6 autocorrelation in EDA). A tree predicting the level
    spends its splits re-learning persistence and returns step-shaped,
    quantised forecasts. Predicting the change makes persistence the built-in
    starting point, so every split is spent on what persistence gets wrong.
    Validation confirmed it: the level model lost to persistence on MAE, the
    change model beat it at every horizon.
    """
    name = "lgbm"

    def __init__(self, **params):
        self.params = dict(
            n_estimators=3000, learning_rate=0.03, num_leaves=31, min_child_samples=100,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
            objective="l1",  # we report MAE; L1 also resists the heavy tails from sudden stops
            random_state=C.SEED, deterministic=True, force_row_wise=True, verbose=-1,
        )
        self.params.update(params)

    def fit(self, X, y, X_val=None, y_val=None):
        self.model_ = lgb.LGBMRegressor(**self.params)
        kw = {}
        if X_val is not None:
            kw = dict(eval_X=(X_val,), eval_y=(y_val - X_val["p_0"],), eval_metric="l1",
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        self.model_.fit(X, y - X["p_0"], **kw)
        return self

    def predict(self, X):
        return np.clip(X["p_0"].to_numpy() + self.model_.predict(X), 0, C.RATED_KW)


def skill_score(rmse_model: float, rmse_ref: float) -> float:
    """Standard wind-forecast skill: 1 - RMSE_model / RMSE_persistence.
    > 0 means the model adds value over doing nothing clever."""
    return 1.0 - rmse_model / rmse_ref
