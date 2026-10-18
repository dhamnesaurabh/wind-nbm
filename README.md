# Wind Turbine Normal-Behaviour Modelling, Underperformance Detection & Short-Term Forecasting

**Avathon AI/ML Hiring Challenge · Track C (Classical ML / Forecasting) · Scenario S4 (Forecasting with anomaly detection), applied to wind energy**
Author: Saurabh Dhamne

> 5-minute walkthrough video: `<(https://www.loom.com/share/6c9e035eb191468abacc43b47d6babab)>`
> Write-up (PDF): [`writeup/Saurabh_Dhamne_TrackC_writeup.pdf`](writeup/Saurabh_Dhamne_TrackC_writeup.pdf)
> Every design decision, alternative and rejection: [`docs/DECISIONS.md`](docs/DECISIONS.md)

## The problem

A wind turbine loses money in two ways operators often cannot see:

1. **It stops.** This is easy to spot.
2. **It keeps running, but produces a few percent less than it should.** Causes include blade soiling, pitch misalignment, icing and sensor drift. Nothing trips an alarm, and a 5% loss on one 3.6 MW turbine costs about **$2,000 per month**.

The reference operators usually compare against, the **manufacturer's power curve, over-predicts this turbine by about 10%**. Measured against it, a simple "below 90% of expected" rule flags **34% of perfectly healthy records**, so the alarms get ignored.

This project builds one physics-informed **normal-behaviour model (NBM)**, the expected power of a *healthy* turbine, and uses it for three things:

| Deliverable | Method | Headline result (test, Nov–Dec 2018) |
|---|---|---|
| Expected healthy power | Monotone LightGBM on wind speed, turbulence, direction | nMAE **2.20%**: 11% better than the IEC binned curve, 33% better than the manufacturer curve |
| Subtle underperformance detection | NBM residual → per-bin standardisation → CUSUM, threshold set for ≤1 false alarm/month | Catches **90% of 10%** losses and **78% of 5%** losses within 7 days. Isolation Forest catches **0%** at the same alarm budget |
| Short-term power forecast (10 min–6 h) | Direct multi-horizon LightGBM predicting the *change* in power | Beats persistence at every horizon, with skill rising from 0.7% to 6.0%. An honest ceiling without weather forecasts |

On clean test data the detector also flagged **three real winter episodes 7–16 hours before the turbine stopped** (consistent with blade icing).

## Data

- **Source:** [Kaggle: Wind Turbine SCADA Dataset](https://www.kaggle.com/datasets/berkerisen/wind-turbine-scada-dataset). It covers one 3.6 MW turbine in Turkey, every 10 minutes through 2018: 50,530 rows of wind speed, direction, active power and theoretical power.
- **Why it fits:**
  - It is real SCADA data, with the messiness a synthetic generator would not invent: 3.9% missing timestamps (gaps up to 104 h), 4.2% stops, a recurring ~3,450 kW derating plateau, and a ~15% seasonal air-density effect.
  - A manufacturer curve is included, so the "what operators do today" baseline is real.
- **Limitations:**
  - One turbine and one year.
  - No temperature, pressure or status codes.
  - No labelled faults, so faults with known ground truth are injected (see `docs/DECISIONS.md` D16).

## Reproduce everything

Requires **Python 3.12 or 3.13** (tested on 3.13). Runs on CPU in about 2 minutes. No GPU is needed.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/download_data.py    # fetches T1.csv and verifies its MD5
python run_all.py                  # every number, table and figure
mlflow ui --backend-store-uri sqlite:///mlflow.db   # optional: browse the experiment log
```

**Outputs:**
- `results/metrics.json`: every metric quoted in the docs
- `results/figures/`: all figures
- `results/mlflow_runs.csv`: the experiment log, 33 runs across 4 experiments

**Determinism:**
- Seeds are fixed (`SEED = 42`).
- LightGBM runs with `deterministic=True`.
- The split is chronological and never shuffled.

## Approach in one picture

```
SCADA (10-min) ──> regular grid + physical state labels (missing / low wind / stopped / operating)
                     │
                     ├─> robust normal-operation filter (train only) ──> NBM: expected healthy power
                     │                                                    │
                     │        ┌───────────────────────────────────────────┤
                     │        ▼                                           ▼
                     │   residual ─> z per wind bin ─> CUSUM ─> alarm    physics feature
                     │                                                    │
                     └─> lag/rolling features ────────────────────────> Δpower forecaster (per horizon)
```

**Split:**
- Train: Jan–Aug
- Validation: Sep–Oct. All model, feature, threshold and `k` choices are made here.
- Test: Nov–Dec. Touched once.

## Repository layout

```
run_all.py                 one command reproduces everything
scripts/download_data.py   dataset fetch + checksum
src/windnbm/
  config.py                every constant, with the reason next to it
  data.py                  load, regular grid, physical state labels, split
  features.py              cyclic encodings, turbulence proxy
  nbm.py                   normal-operation filter + 4 model families
  forecast.py              persistence / Ridge AR / LightGBM-delta
  anomaly.py               standardiser, CUSUM, calibration, Isolation Forest, fault injection
  metrics.py, plots.py, tracking.py (MLflow), pipeline.py (stages)
results/                   metrics.json, figures/, mlflow_runs.csv
docs/DECISIONS.md          decision log: alternatives, rejections, evidence
writeup/                   2-page PDF write-up
```

## Key results

**Normal-behaviour model.** nMAE is MAE as a % of rated power, measured on test points in normal operation.

| Model | Val | Test | Test bias (pred − actual) |
|---|---|---|---|
| Manufacturer curve | 4.10% | 3.27% | +98 kW |
| IEC 61400-12-1 binned curve | 1.93% | 2.46% | −48 kW |
| 4-parameter logistic | 2.26% | 3.10% | −43 kW |
| **LightGBM, monotone [ws, TI, direction]** | **1.80%** | **2.20%** | −45 kW |

**Forecast.** Figures are test nMAE.

| Horizon | Persistence | Ridge AR | LightGBM-Δ |
|---|---|---|---|
| 10 min | 3.49% | 3.72% | **3.48%** |
| 1 h | 7.94% | 9.12% | **7.86%** |
| 3 h | 13.01% | 15.18% | **12.79%** |
| 6 h | 18.69% | 21.37% | **18.04%** |

**Detection.** 7-day injected faults, 40 trials each. A detection counts only if it comes before any alarm on the clean data.

| Fault | CUSUM detection | Median delay | Isolation Forest |
|---|---|---|---|
| Efficiency −3% | 42% | 52 h | 0% |
| Efficiency −5% | 78% | 79 h | 0% |
| Efficiency −10% | 90% | 52 h | 0% |
| Anemometer +5% | 98% | 36 h | 55% |

## Known limitations
- **Seasonal air density is the largest remaining error.** The mean residual (actual − expected) swings from −32 kW in July to +88 kW in February. The fix is temperature data plus the IEC density correction, not more ML.
- **Anemometer drift cannot be told apart from real loss** using one turbine's data alone.
- **Forecasting beyond about 1 hour needs NWP weather forecasts.** Turbine history alone has a low ceiling.
- **One year of data,** so seasonal effects cannot be validated across years.

## Suggested commit sequence
The assignment asks for iterative history. Build the repo in this order, one commit per step:

1. Project skeleton, `config.py`, data download + checksum
2. Data loading, regular grid, physical state labels
3. EDA findings (manufacturer gap, seasonal effect)
4. Normal-operation filter + baseline curves (manufacturer, IEC, logistic)
5. LightGBM NBM + feature ablation + MLflow tracking
6. SHAP explainability + error-by-wind-bin analysis
7. Forecasting: persistence, Ridge, LightGBM level model (lost to persistence)
8. Forecasting: switch to Δ-target, multi-horizon, backtest
9. Anomaly detection: standardiser + CUSUM + calibration
10. Fault injection, k selection on validation, Isolation Forest comparison
11. Rolling-bias ablation (rejected), natural event analysis
12. Business impact, figures, README, decision log, write-up
