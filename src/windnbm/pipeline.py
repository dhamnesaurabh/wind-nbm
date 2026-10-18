"""End-to-end pipeline. Each stage returns what the next one needs and writes
its numbers to results/. Run via `python run_all.py`."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import shap

from . import anomaly as A
from . import config as C
from . import data as D
from . import features as F
from . import forecast as FC
from . import nbm as N
from . import plots as P
from .metrics import regression_metrics
from .tracking import log_run

FEATURE_SETS = {  # ablation: does each piece of context earn its place?
    "ws": ["ws"],
    "ws+ti": ["ws", "ti"],
    "ws+wd": ["ws", "wd_sin", "wd_cos"],
    "ws+hour": ["ws", "hour_sin", "hour_cos"],
    "ws+doy": ["ws", "doy_sin", "doy_cos"],
    "ws+ti+wd": ["ws", "ti", "wd_sin", "wd_cos"],
    "ws+ti+wd+hour": ["ws", "ti", "wd_sin", "wd_cos", "hour_sin", "hour_cos"],
    "all": F.NBM_FEATURES,
}


def _r(x, n=3):
    return None if x is None else round(float(x), n)


# =============================================================== 1. data
def stage_data():
    df = F.build_nbm_frame(D.load_dataset())
    train, val, test = D.split(df)
    summary = {
        "rows_on_grid": len(df),
        "missing_pct": _r(100 * (df["state"] == D.STATE_MISSING).mean(), 2),
        "state_share_pct": {k: _r(100 * v, 2) for k, v in df["state"].value_counts(normalize=True).items()},
        "split_rows": {"train": len(train), "val": len(val), "test": len(test)},
        "split_dates": {"train": [str(train.index.min()), str(train.index.max())],
                        "val": [str(val.index.min()), str(val.index.max())],
                        "test": [str(test.index.min()), str(test.index.max())]},
    }
    p = df["power"]
    summary["power_autocorrelation"] = {f"lag_{k*10}min": _r(p.autocorr(k)) for k in (1, 6, 18, 36, 144)}
    run = df[(df["state"] == D.STATE_OPERATING) & (df["theo"] > 100)]
    summary["actual_over_manufacturer_ratio_median"] = _r((run["power"] / run["theo"]).median())
    summary["negative_power_rows"] = int((D.load_raw()["power"] < 0).sum())
    plateau = df[(df["ws"] > 13) & df["power"].between(3400, 3500)]
    summary["derating_plateau_3400_3500kw_rows_by_month"] = {int(k): int(v) for k, v in plateau.groupby(plateau.index.month).size().items()}
    op = df[(df["state"] == D.STATE_OPERATING) & df["ws"].between(7.5, 8.0)]
    monthly = op["power"].groupby(op.index.month).median()
    summary["median_power_7p5_8ms_by_month"] = {int(k): _r(v, 0) for k, v in monthly.items()}
    P.seasonal_effect(monthly)
    return df, train, val, test, summary


# =============================================================== 2. NBM
def stage_nbm(df, train, val, test):
    filt = N.NormalOperationFilter().fit(train)
    trn, van, ten = (d[filt.mask(d)] for d in (train, val, test))
    te_op = test[test["state"] == D.STATE_OPERATING]
    out = {"filter_kept_pct_of_operating": {
        n: _r(100 * filt.mask(d).sum() / (d["state"] == D.STATE_OPERATING).sum(), 1)
        for n, d in [("train", train), ("val", val), ("test", test)]}}

    def evaluate(model, tag, params):
        m_val = regression_metrics(van["power"], model.predict(van))
        m_test = regression_metrics(ten["power"], model.predict(ten))
        m_all = regression_metrics(te_op["power"], model.predict(te_op))
        log_run("nbm", model.name if tag is None else tag, params,
                {"val_nmae_pct": m_val["nmae_pct"], "val_bias_kw": m_val["bias_kw"],
                 "test_nmae_pct": m_test["nmae_pct"], "test_nrmse_pct": m_test["nrmse_pct"],
                 "test_bias_kw": m_test["bias_kw"], "test_all_operating_nmae_pct": m_all["nmae_pct"]})
        return {"val": {k: _r(v) for k, v in m_val.items()}, "test_normal": {k: _r(v) for k, v in m_test.items()},
                "test_all_operating": {k: _r(v) for k, v in m_all.items()}}

    results, models = {}, {}
    for model in [N.ManufacturerCurve(), N.BinnedCurve(), N.LogisticCurve()]:
        model.fit(trn, trn["power"])
        models[model.name] = model
        results[model.name] = evaluate(model, None, {"family": model.name})
    results["logistic_4p"]["params"] = [_r(p) for p in models["logistic_4p"].params_]

    ablation = {}
    for set_name, feats in FEATURE_SETS.items():
        m = N.LightGBMNBM(features=feats, name=f"lgbm[{set_name}]").fit(trn, trn["power"], van, van["power"])
        models[m.name] = m
        ablation[set_name] = evaluate(m, m.name, {"family": "lightgbm", "features": feats, "monotone": True,
                                                  "best_iteration": m.model_.best_iteration_})
    # Model selection uses VALIDATION only. Test numbers are reported, never used to choose.
    best_set = min(ablation, key=lambda k: ablation[k]["val"]["nmae_pct"])
    chosen = models[f"lgbm[{best_set}]"]
    unc = N.LightGBMNBM(features=FEATURE_SETS[best_set], monotone=False,
                        name=f"lgbm[{best_set}]-unconstrained").fit(trn, trn["power"], van, van["power"])
    ablation[f"{best_set} (no monotone constraint)"] = evaluate(unc, unc.name, {"family": "lightgbm", "features": FEATURE_SETS[best_set], "monotone": False})
    out.update(baselines=results, lightgbm_ablation=ablation, chosen_feature_set=best_set,
               chosen_features=FEATURE_SETS[best_set])

    # Bias by month of the chosen model on normal data - the seasonal story
    allnorm = pd.concat([trn, van, ten])
    res = allnorm["power"] - chosen.predict(allnorm)
    pl = allnorm["ws"] < C.PARTIAL_LOAD_MAX_MS
    out["chosen_bias_by_month_partial_load_kw"] = {int(k): _r(v, 0) for k, v in res[pl].groupby(allnorm.index[pl].month).mean().items()}

    # Error by wind-speed bin (test, normal operation)
    b = np.floor(ten["ws"]).clip(upper=16)
    tab = pd.DataFrame({
        "manufacturer curve": (models["manufacturer_curve"].predict(ten) - ten["power"]).abs().groupby(b).mean(),
        "IEC binned curve": (models["binned_iec"].predict(ten) - ten["power"]).abs().groupby(b).mean(),
        "LightGBM (chosen)": (chosen.predict(ten) - ten["power"]).abs().groupby(b).mean(),
    })
    tab = tab[tab.index >= 3]
    P.error_by_ws(tab)
    out["test_mae_by_ws_bin_kw"] = tab.round(0).to_dict()

    # Explainability
    sample = trn.sample(3000, random_state=C.SEED)[chosen.features]
    sv = shap.TreeExplainer(chosen.model_).shap_values(sample)
    imp = np.abs(sv).mean(axis=0)
    out["shap_mean_abs_kw"] = {f: _r(v, 1) for f, v in zip(chosen.features, imp)}
    P.shap_bar(chosen.features, imp, "What drives expected power (normal-behaviour model)", "f4_shap_nbm.png")
    # directional effect of wind direction, for the domain story
    wdir = np.degrees(np.arctan2(sample["wd_sin"], sample["wd_cos"])) % 360 if "wd_sin" in sample else None
    if wdir is not None:
        contrib = pd.Series(sv[:, chosen.features.index("wd_sin")] + sv[:, chosen.features.index("wd_cos")], index=sample.index)
        out["shap_direction_by_sector_kw"] = {f"{int(s*45)}-{int(s*45+45)}": _r(v, 0)
                                             for s, v in contrib.groupby((wdir // 45).astype(int)).mean().items()}

    # Power-curve figure
    ws = np.linspace(0, 20, 200)
    grid = pd.DataFrame({"ws": ws})
    for f in chosen.features:
        if f != "ws":
            grid[f] = float(trn[f].median())
    theo = df.dropna(subset=["theo"]).sort_values("ws")
    P.power_curve(test, {
        "manufacturer curve": (theo["ws"].to_numpy(), theo["theo"].to_numpy()),
        "IEC binned curve (train)": (ws, models["binned_iec"].predict(grid)),
        "LightGBM, median conditions": (ws, chosen.predict(grid)),
    })
    return chosen, filt, models, out


# =============================================================== 3. forecasting
def _expected_full(model, df):
    return N.predict_expected(model, df)


def stage_forecast(df, nbm):
    X = FC.build_forecast_features(df, _expected_full(nbm, df))
    out, nmae_tab, models_1h = {}, {}, {}
    for hname, h in C.HORIZONS.items():
        y = FC.make_target(df, h)
        ok = FC.complete_rows(X, y)
        target_time = df.index + pd.Timedelta(minutes=10 * h)
        # the TARGET must also sit inside the split: no label from the future fold
        i_tr = ok & (target_time < pd.Timestamp(C.TRAIN_END))
        i_va = ok & (df.index >= C.TRAIN_END) & (target_time < pd.Timestamp(C.VAL_END))
        i_te = ok & (df.index >= C.VAL_END)
        res = {}
        for m in [FC.Persistence(), FC.RidgeAR(), FC.LightGBMForecaster()]:
            m.fit(X[i_tr], y[i_tr], X[i_va], y[i_va])
            res[m.name] = {"val": regression_metrics(y[i_va], m.predict(X[i_va])),
                           "test": regression_metrics(y[i_te], m.predict(X[i_te]))}
            if hname == "1h":
                models_1h[m.name] = m
        ref_v, ref_t = res["persistence"]["val"]["rmse_kw"], res["persistence"]["test"]["rmse_kw"]
        for name, r in res.items():
            r["val"]["skill"] = FC.skill_score(r["val"]["rmse_kw"], ref_v)
            r["test"]["skill"] = FC.skill_score(r["test"]["rmse_kw"], ref_t)
            log_run("forecast", f"{name}@{hname}", {"model": name, "horizon_steps": h},
                    {"val_nmae_pct": r["val"]["nmae_pct"], "val_skill": r["val"]["skill"],
                     "test_nmae_pct": r["test"]["nmae_pct"], "test_skill": r["test"]["skill"]})
        out[hname] = {n: {s: {k: _r(v) for k, v in r[s].items()} for s in r} for n, r in res.items()}
        nmae_tab[hname] = {n: r["test"]["nmae_pct"] for n, r in res.items()}
    tab = pd.DataFrame(nmae_tab).T
    P.forecast_skill(tab)

    # Expanding-window backtest at 1 h, months Apr-Dec. Iterations fixed from the
    # main 1h model so the backtest needs no extra validation split.
    h = C.HORIZONS["1h"]
    y = FC.make_target(df, h)
    ok = FC.complete_rows(X, y)
    n_iter = models_1h["lgbm"].model_.best_iteration_ or 500
    bt = {}
    for month in range(4, 13):
        start = pd.Timestamp(2018, month, 1)
        end = start + pd.offsets.MonthBegin(1)
        tgt = df.index + pd.Timedelta(minutes=10 * h)
        i_tr = ok & (tgt < start)
        i_te = ok & (df.index >= start) & (df.index < end)
        lg = FC.LightGBMForecaster(n_estimators=n_iter).fit(X[i_tr], y[i_tr])
        bt[start.strftime("%b")] = {
            "persistence": regression_metrics(y[i_te], X.loc[i_te, "p_0"])["nmae_pct"],
            "lgbm": regression_metrics(y[i_te], lg.predict(X[i_te]))["nmae_pct"],
        }
    btab = pd.DataFrame(bt).T
    P.backtest(btab)
    out["backtest_1h_nmae_pct"] = btab.round(3).to_dict()
    out["backtest_1h_months_lgbm_better"] = int((btab["lgbm"] < btab["persistence"]).sum())

    # Explainability of the 1 h forecaster
    lg = models_1h["lgbm"]
    i_te = ok & (df.index >= C.VAL_END)
    samp = X[i_te].sample(2000, random_state=C.SEED)
    sv = shap.TreeExplainer(lg.model_).shap_values(samp)
    imp = pd.Series(np.abs(sv).mean(axis=0), index=X.columns).sort_values(ascending=False).head(10)
    out["shap_1h_top10_kw"] = {k: _r(v, 1) for k, v in imp.items()}
    P.shap_bar(list(imp.index), imp.values, "What drives the 1-hour forecast correction", "f8_shap_forecast_1h.png")
    return out


# =============================================================== 4. anomaly detection
class Detector:
    """NBM -> standardised residual -> CUSUM. Bundled so injection runs reuse it."""

    def __init__(self, nbm, std, k, h):
        self.nbm, self.std, self.k, self.h = nbm, std, k, h

    def run(self, d):
        e = N.predict_expected(self.nbm, d)
        z = self.std.transform(d["ws"], d["power"] - e)
        s, a = A.cusum_lower(z, A.eligible(d), self.k, self.h)
        return e, z, s, a


def _event_starts(alarm: pd.Series, gap_h=24):
    t = alarm[alarm].index
    if len(t) == 0:
        return []
    starts = [t[0]]
    for prev, cur in zip(t[:-1], t[1:]):
        if cur - prev > pd.Timedelta(hours=gap_h):
            starts.append(cur)
    return starts


def _inject_runs(det, base_df, starts, kind, mag, dur_h, slice_to=None):
    """Return the CUSUM alarm series for each injected copy of base_df."""
    runs = []
    for s in starts:
        d = A.inject_fault(base_df, s, dur_h, kind, mag)
        a = det.run(d)[3]
        runs.append(a if slice_to is None else a.loc[slice_to])
    return runs


def stage_anomaly(df, train, val, test, nbm, filt):
    out = {}
    dur_h = 24 * 7  # a real fault persists until repaired; we ask "caught within a week?"
    van = val[filt.mask(val)]
    std = A.ResidualStandardiser().fit(van["ws"], van["power"] - N.predict_expected(nbm, van))
    z_val = std.transform(val["ws"], val["power"] - N.predict_expected(nbm, val))
    m_val = filt.mask(val)
    grid = np.arange(1, 600, 1.0)

    # ---- design choice 1: CUSUM reference value k, chosen by injecting faults into VALIDATION
    rng = np.random.default_rng(C.SEED + 1)
    v_starts = pd.to_datetime(rng.uniform(pd.Timestamp("2018-09-01").value,
                                          pd.Timestamp("2018-10-24").value, 40)).floor("10min")
    k_table = {}
    for k in [0.25, 0.5, 1.0, 2.0]:
        h_k = A.calibrate_h(z_val, m_val, k, C.FALSE_ALARMS_PER_30D, grid=grid)
        det_k = Detector(nbm, std, k, h_k)
        base = det_k.run(val)[3]
        row = {"h": h_k}
        for mag in (0.05, 0.10):
            rate, dl = A.credited_detections(base, _inject_runs(det_k, val, v_starts, "efficiency_loss", mag, dur_h), v_starts, dur_h)
            row[f"val_detect_{int(mag*100)}pct"] = _r(rate)
            row[f"val_delay_{int(mag*100)}pct_h"] = _r(np.median(dl), 1) if dl else None
        row["score"] = (row["val_detect_5pct"] + row["val_detect_10pct"]) / 2
        k_table[k] = row
        log_run("anomaly-design", f"cusum k={k}", {"k": k, "h": h_k}, {kk: v for kk, v in row.items() if isinstance(v, float)})
    k_best = max(k_table, key=lambda kk: k_table[kk]["score"])
    out["k_selection_on_validation"] = {str(kk): v for kk, v in k_table.items()}

    h = A.calibrate_h(z_val, m_val, k_best, C.FALSE_ALARMS_PER_30D, grid=grid)
    det = Detector(nbm, std, k_best, h)
    a_val = det.run(val)[3]
    out["cusum"] = {"k": k_best, "h": h, "z_clip": A.Z_CLIP,
                    "val_alarms_per_30d_all_operating": _r(A.alarms_per_30d(a_val), 2),
                    "residual_scale_by_ws_bin_kw": {int(b): _r(v, 0) for b, v in std.scale_.items()}}

    # ---- design choice 2 (ablation): rolling 14-day bias correction for seasonal drift
    z_full = std.transform(df["ws"], df["power"] - N.predict_expected(nbm, df))
    el_full = A.eligible(df)
    vs = slice(C.TRAIN_END, pd.Timestamp(C.VAL_END) - pd.Timedelta(minutes=10))
    z_adj = A.rolling_bias_adjust(z_full, el_full)
    h_adj = A.calibrate_h(z_adj.loc[vs], m_val, k_best, C.FALSE_ALARMS_PER_30D, grid=grid)
    base_adj = A.cusum_lower(z_adj.loc[vs], el_full.loc[vs], k_best, h_adj)[1]
    runs_adj = []
    for s in v_starts:
        d = A.inject_fault(df, s, dur_h, "efficiency_loss", 0.05)
        z = A.rolling_bias_adjust(std.transform(d["ws"], d["power"] - N.predict_expected(nbm, d)), A.eligible(d))
        runs_adj.append(A.cusum_lower(z.loc[vs], A.eligible(d).loc[vs], k_best, h_adj)[1])
    rate_adj, _ = A.credited_detections(base_adj, runs_adj, v_starts, dur_h)
    out["rolling_bias_ablation_val_detect_5pct"] = {"without": k_table[k_best]["val_detect_5pct"], "with": _r(rate_adj)}

    # ---- comparator, same alarm budget
    ifd = A.IsolationForestDetector().fit(train)
    tau = ifd.calibrate(ifd.score(val).where(m_val), C.FALSE_ALARMS_PER_30D)
    out["isolation_forest"] = {"tau": _r(tau, 4), "window_points": ifd.window, "frac": ifd.frac}

    # ---- clean test run: natural events
    e_te, z_te, s_te, a_te = det.run(test)
    a_if_te = ifd.alarms(ifd.score(test), tau)
    out["test_clean_alarm_events_cusum"] = [str(t) for t in _event_starts(a_te)]
    out["test_clean_alarm_events_iforest"] = [str(t) for t in _event_starts(a_if_te)]

    # lead time: did the CUSUM alarm before a long stop (>= 3 h)?
    stopped = (test["state"] == D.STATE_STOPPED).astype(int)
    grp = (stopped != stopped.shift()).cumsum()
    runs = stopped.groupby(grp).agg(["first", "size"])
    stops = []
    for g, r in runs[(runs["first"] == 1) & (runs["size"] >= 18)].iterrows():
        t0 = stopped[grp == g].index[0]
        prior = a_te[(a_te.index < t0) & (a_te.index >= t0 - pd.Timedelta(hours=48))]
        first = prior[prior].index.min() if prior.any() else None
        stops.append({"stop_start": str(t0), "stop_hours": _r(r["size"] / 6, 1),
                      "first_alarm_within_48h_before": None if first is None else str(first),
                      "lead_time_h": None if first is None else _r((t0 - first).total_seconds() / 3600, 1)})
    out["test_long_stops_and_early_warning"] = stops

    for i, (a, b) in enumerate([("2018-12-18", "2018-12-21"), ("2018-12-24", "2018-12-27")]):
        d = pd.DataFrame({"power": test["power"], "expected": e_te}).loc[a:b]
        P.event_panel(d, s_te.loc[a:b], h, a_te.loc[a:b],
                      f"Natural event {a[5:]}: power drifts below expected, then the turbine stops",
                      f"f9{'ab'[i]}_event_{a[5:]}.png")

    # ---- final evaluation: fault injection on TEST (touched once, design frozen)
    rng = np.random.default_rng(C.SEED)
    starts = pd.to_datetime(rng.uniform(pd.Timestamp("2018-11-01").value,
                                        pd.Timestamp("2018-12-24").value, 40)).floor("10min")
    scenarios = [("efficiency_loss", 0.03), ("efficiency_loss", 0.05), ("efficiency_loss", 0.08),
                 ("efficiency_loss", 0.10), ("anemometer_drift", 0.05)]
    inj, example = {}, None
    for kind, mag in scenarios:
        cus_runs, if_runs = [], []
        for s in starts:
            d = A.inject_fault(test, s, dur_h, kind, mag)
            e_i, _, s_i, a_i = det.run(d)
            cus_runs.append(a_i)
            if_runs.append(ifd.alarms(ifd.score(d), tau))
            if example is None and kind == "efficiency_loss" and mag == 0.08:
                w = (a_i.index >= s) & (a_i.index < s + pd.Timedelta(hours=dur_h))
                if a_i[w].any() and not a_te[w].any():
                    example = (s, d, e_i, s_i, a_i)
        r_c, d_c = A.credited_detections(a_te, cus_runs, starts, dur_h)
        r_i, d_i = A.credited_detections(a_if_te, if_runs, starts, dur_h)
        label = f"{kind.replace('_', ' ')} {int(mag*100)}%"
        inj[label] = {"trials": len(starts),
                      "cusum_detection_rate": _r(r_c), "cusum_median_delay_h": _r(np.median(d_c), 1) if d_c else None,
                      "iforest_detection_rate": _r(r_i), "iforest_median_delay_h": _r(np.median(d_i), 1) if d_i else None}
        log_run("anomaly", label, {"kind": kind, "magnitude": mag, "duration_h": dur_h, "k": k_best, "h": h},
                {"cusum_detection_rate": r_c, "iforest_detection_rate": r_i})
    out["fault_injection_7d"] = inj
    tab = pd.DataFrame({lbl: {"NBM residual + CUSUM": r["cusum_detection_rate"], "Isolation Forest": r["iforest_detection_rate"]}
                        for lbl, r in inj.items()}).T
    P.detection_rates(tab)
    if example is not None:
        s, d, e_i, s_i, a_i = example
        lo, hi = s - pd.Timedelta(days=1), s + pd.Timedelta(days=5)
        dd = pd.DataFrame({"power": d["power"], "expected": e_i}).loc[lo:hi]
        P.event_panel(dd, s_i.loc[lo:hi], h, a_i.loc[lo:hi], "Injected 8% efficiency loss: CUSUM accumulates evidence, then alarms",
                      "f10_injected_example.png", fault_start=s)
    return det, out


# =============================================================== 5. business impact
def stage_impact(test, nbm, filt, forecast_out, models):
    e = N.predict_expected(nbm, test)
    windy = test["ws"] >= C.CUT_IN_MS
    to_mwh = 1 / 6 / 1000  # 10-min kW samples -> MWh
    st = test["state"]
    normal = filt.mask(test)
    lost_stop = e[windy & (st == D.STATE_STOPPED)].sum() * to_mwh
    abn = (st == D.STATE_OPERATING) & ~normal
    lost_abn = (e - test["power"])[abn].clip(lower=0).sum() * to_mwh
    expected_total = e[windy & st.isin([D.STATE_OPERATING, D.STATE_STOPPED])].sum() * to_mwh
    actual_total = test["power"][st.isin([D.STATE_OPERATING, D.STATE_STOPPED, D.STATE_LOW_WIND])].sum() * to_mwh
    usd = 1000 * C.TARIFF_USD_PER_KWH
    out = {
        "period": "test, 2018-11-01 to 2018-12-31",
        "expected_energy_mwh": _r(expected_total, 1), "actual_energy_mwh": _r(actual_total, 1),
        "lost_in_stops_mwh": _r(lost_stop, 1), "lost_in_stops_usd": _r(lost_stop * usd, 0),
        "lost_in_abnormal_running_mwh": _r(lost_abn, 1), "lost_in_abnormal_running_usd": _r(lost_abn * usd, 0),
        "tariff_usd_per_kwh": C.TARIFF_USD_PER_KWH,
    }
    # Why not just use the manufacturer curve? Count false flags of a naive rule.
    ten = test[normal]
    for name, pred in [("manufacturer_curve", models["manufacturer_curve"].predict(ten)), ("chosen_nbm", nbm.predict(ten))]:
        out[f"naive_rule_flags_pct_on_healthy_points[{name}]"] = _r(100 * (ten["power"] < 0.9 * pred).mean(), 1)
    # What does an undetected 5% partial-load loss cost per month?
    pl = (st == D.STATE_OPERATING) & (test["ws"] < C.PARTIAL_LOAD_MAX_MS)
    pl_mwh_month = test["power"][pl].sum() * to_mwh / 2
    out["partial_load_energy_mwh_per_month"] = _r(pl_mwh_month, 1)
    out["cost_of_5pct_partial_load_loss_usd_per_month"] = _r(0.05 * pl_mwh_month * usd, 0)
    # Forecast value: absolute error avoided vs persistence over the test period
    for hname in ["1h", "6h"]:
        r = forecast_out[hname]
        dmae_kw = (r["persistence"]["test"]["mae_kw"] - r["lgbm"]["test"]["mae_kw"])
        hours = r["lgbm"]["test"]["n"] / 6
        out[f"forecast_error_avoided_mwh[{hname}]"] = _r(dmae_kw * hours / 1000, 1)
    return out


def run_all():
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    df, train, val, test, data_summary = stage_data()
    print("[1/5] data ready", data_summary["split_rows"])
    nbm, filt, models, nbm_out = stage_nbm(df, train, val, test)
    print("[2/5] NBM chosen:", nbm_out["chosen_feature_set"])
    fc_out = stage_forecast(df, nbm)
    print("[3/5] forecasting done")
    _, an_out = stage_anomaly(df, train, val, test, nbm, filt)
    print("[4/5] anomaly detection done, h =", an_out["cusum"]["h"])
    impact = stage_impact(test, nbm, filt, fc_out, models)
    print("[5/5] business impact done")
    results = {"data": data_summary, "nbm": nbm_out, "forecast": fc_out, "anomaly": an_out, "impact": impact}
    (C.RESULTS / "metrics.json").write_text(json.dumps(results, indent=2, default=str))
    return results
