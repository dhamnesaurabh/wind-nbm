"""Normal Behaviour Model (NBM): expected power of a HEALTHY turbine.

Four candidate families, from zero-training to multivariate ML:
  1. ManufacturerCurve - the curve shipped with the turbine (no training)
  2. BinnedCurve       - IEC 61400-12-1 "method of bins" (industry standard)
  3. LogisticCurve     - 4-parameter physics-shaped parametric fit
  4. LightGBM          - multivariate gradient boosting, monotone in wind speed
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from . import config as C
from .data import STATE_OPERATING
from .features import NBM_FEATURES


# ====================================================================== filter
class NormalOperationFilter:
    """Decide which training rows represent healthy, unconstrained operation.

    Two layers:
      * physical state rules (stopped / low wind / missing) - from data.py
      * robust per-wind-bin outlier rejection: keep a point if
        |power - bin median| <= k * max(1.4826 * MAD, floor)
    Median/MAD instead of mean/std because the outliers we remove (derating
    plateaus, partial stops) would themselves inflate a std-based band.
    """

    def __init__(self, k=C.FILTER_K, bin_width=C.BIN_WIDTH_MS, min_scale=C.FILTER_MIN_SCALE_KW):
        self.k, self.bin_width, self.min_scale = k, bin_width, min_scale

    def _bin(self, ws):
        return np.floor(ws / self.bin_width).astype("Int64")

    def fit(self, df: pd.DataFrame):
        op = df[df["state"] == STATE_OPERATING]
        b = self._bin(op["ws"])
        grp = op["power"].groupby(b)
        self.median_ = grp.median()
        mad = grp.apply(lambda s: np.median(np.abs(s - s.median())))
        self.scale_ = np.maximum(1.4826 * mad, self.min_scale)
        return self

    def mask(self, df: pd.DataFrame) -> pd.Series:
        b = self._bin(df["ws"])
        med = b.map(self.median_).astype(float)
        scale = b.map(self.scale_).astype(float)
        ok = (df["power"] - med).abs() <= self.k * scale
        return (df["state"] == STATE_OPERATING) & ok.fillna(False)


# ====================================================================== models
class ManufacturerCurve:
    """Baseline 0: the theoretical curve in the dataset. Zero training."""
    name = "manufacturer_curve"

    def fit(self, X, y):
        return self

    def predict(self, X):
        return X["theo"].to_numpy()


class BinnedCurve:
    """Baseline 1: IEC 61400-12-1 method of bins.

    Mean power per 0.5 m/s bin, linearly interpolated. It is what a wind
    performance engineer would build first, so ML must beat it to be worth it.
    Limitation: one input (wind speed) - blind to season, direction, turbulence.
    """
    name = "binned_iec"

    def __init__(self, bin_width=C.BIN_WIDTH_MS, min_count=5):
        self.bin_width, self.min_count = bin_width, min_count

    def fit(self, X, y):
        b = np.floor(X["ws"] / self.bin_width)
        g = pd.DataFrame({"ws": X["ws"], "p": y, "b": b}).groupby("b")
        tab = g.agg(ws=("ws", "mean"), p=("p", "mean"), n=("p", "size"))
        tab = tab[tab["n"] >= self.min_count]
        self.ws_, self.p_ = tab["ws"].to_numpy(), tab["p"].to_numpy()
        return self

    def predict(self, X):
        p = np.interp(X["ws"], self.ws_, self.p_, left=0.0, right=C.RATED_KW)
        return np.where(X["ws"] < C.CUT_IN_MS, 0.0, p)


def _logistic(v, p_max, k, v0, g):
    return p_max / np.power(1.0 + np.exp(-k * (v - v0)), g)


class LogisticCurve:
    """Baseline 2: 4-parameter generalised logistic, P(v) = Pmax/(1+e^-k(v-v0))^g.

    Smooth, monotone, every parameter has a physical meaning (rated power,
    steepness, inflection wind speed, asymmetry), and it extrapolates sanely.
    Weakness: a fixed shape cannot bend to site-specific quirks.
    """
    name = "logistic_4p"

    def fit(self, X, y):
        p0 = [C.RATED_KW, 0.8, 8.5, 1.0]
        bounds = ([0.8 * C.RATED_KW, 0.05, 3.0, 0.1], [1.1 * C.RATED_KW, 5.0, 15.0, 10.0])
        self.params_, _ = curve_fit(_logistic, X["ws"], y, p0=p0, bounds=bounds, maxfev=20000)
        return self

    def predict(self, X):
        p = _logistic(X["ws"].to_numpy(), *self.params_)
        return np.where(X["ws"] < C.CUT_IN_MS, 0.0, p)


class LightGBMNBM:
    """Main model: gradient-boosted trees on wind speed + context.

    monotone=True forces predicted power to be non-decreasing in wind speed
    (all else equal). That is physics we KNOW below cut-out, so we put it in as
    a constraint instead of hoping the model learns it - it removes spurious
    wiggles in sparse high-wind bins, where a wiggle would become a false alarm.
    """

    def __init__(self, features=NBM_FEATURES, monotone=True, name=None, **params):
        self.features = list(features)
        self.monotone = monotone
        self.name = name or ("lgbm_monotone" if monotone else "lgbm_unconstrained")
        self.params = dict(
            n_estimators=2000, learning_rate=0.03, num_leaves=31, min_child_samples=50,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.9, reg_lambda=1.0,
            random_state=C.SEED, deterministic=True, force_row_wise=True, verbose=-1,
        )
        self.params.update(params)

    def fit(self, X, y, X_val=None, y_val=None):
        params = dict(self.params)
        if self.monotone:
            params["monotone_constraints"] = [1 if f == "ws" else 0 for f in self.features]
            params["monotone_constraints_method"] = "advanced"
        self.model_ = lgb.LGBMRegressor(**params)
        fit_kw = {}
        if X_val is not None:
            # Early stopping on the validation months - never on test.
            fit_kw = dict(eval_X=(X_val[self.features],), eval_y=(y_val,), eval_metric="l1",
                          callbacks=[lgb.early_stopping(100, verbose=False)])
        self.model_.fit(X[self.features], y, **fit_kw)
        return self

    def predict(self, X):
        p = self.model_.predict(X[self.features])
        # The controller caps output at rated power; a prediction above it can
        # only be extrapolation noise, and would show up as a fake shortfall.
        return np.clip(p, 0.0, C.RATED_KW)


def predict_expected(model, df: pd.DataFrame) -> pd.Series:
    """Expected healthy power for every row where the model's inputs exist."""
    feats = getattr(model, "features", ["ws"])
    need = list(dict.fromkeys(feats + ["ws"] + (["theo"] if isinstance(model, ManufacturerCurve) else [])))
    ok = df[need].notna().all(axis=1)
    out = pd.Series(np.nan, index=df.index, name="expected")
    if ok.any():
        out[ok] = model.predict(df[ok])
    return out
