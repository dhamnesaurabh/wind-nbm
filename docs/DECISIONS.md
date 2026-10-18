# Decision log: why every choice was made

Every significant decision in this project, the alternatives considered, why they were rejected, and the evidence. Numbers come from `results/metrics.json` (reproduced by `python run_all.py`).

**Rule followed throughout:** a design choice is made on **validation** data (Sep–Oct). The **test** months (Nov–Dec) are used once, at the end, to report results. When test numbers would have favoured a different choice, I say so openly rather than switching (see D7).

---

## A. Problem framing

### D1. Scenario S4 (forecasting + anomaly detection), applied to wind energy
- **Decision:** Treat a wind turbine as the "business" and its electricity as the revenue to forecast and protect. Three linked deliverables:
  1. A **normal-behaviour model (NBM)**: the power a healthy turbine *should* produce.
  2. **Underperformance detection** on the NBM residual.
  3. A **short-term power forecast** (10 min to 6 h).
- **Why this domain:**
  - Energy is a core Avathon vertical, and NBM-based asset monitoring is a real, revenue-linked problem.
  - Turbines lose energy silently through blade soiling, pitch misalignment, icing and sensor drift. A turbine that runs at 95% for a month never trips an alarm, but it loses real money.
- **Why one NBM feeds both tasks:** the "expected healthy power" is the shared asset.
  - In detection, it defines what "abnormal" means.
  - In forecasting, it is a physics feature that tells the model whether low power is due to low wind or to a stopped turbine.

### D2. Track C, not A, B or D
- **The nature of the data:** structured 10-minute sensor data with a strong physical law (P ∝ ½ρAC<sub>p</sub>v³). The questions are numeric: how much power, and is it abnormal?
- **Why not an LLM track:** LLMs add nothing to predicting a number from five sensor columns.
  - RAG (D) and fine-tuning (B) are document and language tools.
  - Agents (A) would orchestrate decisions that are not yet trustworthy; the detection must be right before anything should act on it.
- **Where GenAI fits later:** an agent that turns a CUSUM alarm into a work order. That only works once the alarm itself is reliable, and making it reliable is this project.

### D3. Dataset: Kaggle "Wind Turbine SCADA Dataset" (Turkey, 2018)
- **What it is:** 50,530 real 10-minute records from one 3.6 MW turbine (wind speed, direction, active power, and the manufacturer's theoretical power).
- **Why real data over synthetic:** real SCADA has the messiness a synthetic generator would not invent. That includes 3.9% missing timestamps, 4.2% stops, a recurring derating plateau at ~3,450 kW, negative idle power, and a seasonal air-density effect.
- **Limitations, acknowledged:**
  - One turbine, one year.
  - No temperature or pressure, so air density cannot be computed.
  - No status or alarm codes.
  - No labelled faults, which is why D16 creates ground truth.
- **Integrity:** two independent mirrors were checked byte-identical (MD5 `61655b92…`). `scripts/download_data.py` verifies the hash.

---

## B. Data preparation

### D4. Re-index onto a complete 10-minute grid; never interpolate
- **The risk:** 2,030 timestamps are missing, some in gaps of up to 104 hours. On the raw file, `shift(6)` ("1 hour ago") would silently jump across a gap.
- **Decision:** re-index onto a full grid. Missing rows become NaN and are labelled `missing`.
- **Why not interpolate:** an interpolated value looks like a real measurement to every model. Across a 4-day gap it would invent four days of fake wind.

### D5. Label operating state from physics rules, not from a model
- **States:**
  - `low_wind`: wind speed below 3.0 m/s, the cut-in speed read from the manufacturer curve.
  - `stopped`: wind speed at least 3.5 m/s and power at most 1 kW. The 0.5 m/s margin avoids labelling start-up hysteresis as a stop.
  - `operating`: everything else with data.
- **Why rules:** the labels must be independent of the anomaly detector they are later used to evaluate. Otherwise the evaluation is circular.
- **Negative power** (57 rows, down to −2.5 kW) is clipped to 0. It is the idle turbine drawing auxiliary power, which is physically real but irrelevant to production.

### D6. Chronological split: train Jan–Aug, validation Sep–Oct, test Nov–Dec
- **Why chronological:** a random split leaks the future. Neighbouring 10-minute records are 98% correlated, so a random split would test the model on near-copies of its training rows.
- **Why these months:**
  - Training includes both winter (Jan–Mar) and summer (Jun–Aug), so the model sees both air densities.
  - Validation and test are two months each, so every reported number covers about 8,800 records.
- **Forecasting adds a stricter rule:** the *target* timestamp (t + h) must also fall inside the split, so no training label comes from the validation months.
- **Rejected alternative:** rolling-origin cross-validation for the NBM. With only one year of data, every fold would be a different season, which confounds model quality with seasonality. Rolling-origin is used where it is informative: the forecasting backtest (D13).

---

## C. Normal-behaviour model

### D7. Train the NBM only on "normal" data, found with a robust filter
- **Why filter:** an NBM must learn healthy behaviour. Training on stops and derates teaches it that faults are normal.
- **The filter:** per 0.5 m/s wind bin (the IEC 61400-12-1 bin width), keep a point if |power − bin median| ≤ 3 × max(1.4826·MAD, 18 kW).
- **Why median/MAD and not mean/std:** the outliers being removed (derating plateaus, partial stops) would inflate a standard deviation and hide themselves.
- **Why an 18 kW floor:** at rated power the MAD is close to 0, so without a floor normal ±20 kW jitter would be rejected.
- **What it keeps:** 94.3% of operating points in train, 98.6% in validation, 88.4% in test. December contains real events (see D18).
- **Known bias, stated:** test accuracy is measured on points that pass this filter, which slightly favours models close to the training median curve. I therefore also report error on *all* operating test points (`test_all_operating`). The ranking does not change.

### D8. Four model families compared, including two strong simple baselines

| Model | Why it is a candidate | Val nMAE | Test nMAE | Test bias* |
|---|---|---|---|---|
| Manufacturer curve | What operators use today. Zero training. | 4.10% | 3.27% | +98 kW |
| IEC binned curve | Industry standard (method of bins). Must be beaten for ML to be worth it. | 1.93% | 2.46% | −48 kW |
| 4-parameter logistic | Physics-shaped, interpretable parameters, extrapolates sanely | 2.26% | 3.10% | −43 kW |
| **LightGBM, monotone [ws, TI, direction]** | Uses context the curves cannot see | **1.80%** | **2.20%** | −45 kW |

\*bias = predicted − actual, on test normal points. nMAE = MAE as a % of 3,600 kW rated power.

- **The finding that motivates the whole project:** the manufacturer curve over-predicts by about 10% (median actual/theoretical ratio is 0.90). Using it as the reference, a simple "alarm below 90% of expected" rule would flag **34% of healthy test points**, versus 8.5% with the learned NBM.
- **LightGBM vs the IEC curve:** about 11% lower error on test (2.20% vs 2.46%) and 7% lower on validation. The gain is concentrated in the steep 8–11 m/s region, where a small wind error means a large power error (figure f3).
- **Why gradient boosting rather than the alternatives:**
  - **Neural network or MLP:** about 25,000 rows and 4 tabular features. Gradient-boosted trees are the stronger default for small tabular data, train in seconds, and work with SHAP. A NN would add tuning cost and opacity for no expected gain.
  - **LSTM:** the NBM is an *instantaneous* mapping (conditions now → power now). A sequence model would let past power leak in, and D10 forbids that.
  - **Gaussian process:** popular for power curves and gives uncertainty, but costs O(n³) with 25,000 points. A sparse GP is the natural upgrade if prediction intervals become a requirement.
  - **Random forest:** similar accuracy, but no monotone constraints (D9) and slower.
  - **Polynomial regression:** oscillates at the ends of the range (Runge) and cannot flatten at rated power.
- **Logistic model trade-off:** it has a clean physical reading (P<sub>max</sub> 3,727 kW, steepness 0.57, inflection 8.3 m/s, asymmetry 1.28), but a fixed shape cannot follow this turbine's real curve. That is the price of interpretability, visible in its higher error.

### D9. Monotone constraint on wind speed
- **The constraint:** expected power must not decrease as wind speed increases, all else equal. That is physics below cut-out.
- **Evidence:** accuracy is almost unchanged (validation 1.795% constrained vs 1.807% unconstrained).
- **Why keep it anyway:** it guarantees physically valid curves where data is thin. The unconstrained model can learn small dips in sparse high-wind bins, and in a detector every such dip becomes a false alarm.
- I report this honestly as a **robustness** choice, not an accuracy gain.

### D10. No lagged power in the NBM
- **Why:** if the model can see power from 10 minutes ago, it learns to copy it. When a fault starts, the "expected" value follows the faulty turbine down and the residual disappears.
- **Takeaway:** an NBM should only see *conditions* (wind, direction, turbulence), never the output it is meant to judge.

### D11. Feature ablation decides the inputs, on validation

| Feature set | Val nMAE | Test nMAE (reported, not used) |
|---|---|---|
| ws | 1.919 | 2.448 |
| ws + TI | 1.893 | 2.406 |
| ws + direction | 1.807 | 2.227 |
| ws + hour | 1.822 | 2.459 |
| ws + day-of-year | 2.119 | **2.145** |
| **ws + TI + direction** (chosen) | **1.795** | 2.199 |
| ws + TI + direction + hour | 1.863 | 2.291 |
| all 8 features | 2.526 | 3.067 |

- **Direction helps** (terrain and wake effects). SHAP shows the dominant 45–90° wind sector produces about 28 kW less than average at the same wind speed.
- **Turbulence helps a little.** Gusty wind yields less energy than steady wind of the same 10-minute mean.
- **Day-of-year: the most interesting result in the project.**
  - Season clearly matters physically. At 7.5–8 m/s the median power is 1,392 kW in February but 1,177 kW in July, about 15%, which is the effect of cold, dense air (figure f2).
  - Day-of-year *hurts* on validation but would *help* on test. With one year of data, Sep–Oct has no "same season" in training, while Nov–Dec can borrow from Jan–Feb.
  - **I did not switch to it**, because choosing on test would be leakage.
  - **Lesson for production:** the right fix is not a calendar proxy but the real variable. Add ambient temperature and pressure, and apply the IEC air-density correction v<sub>norm</sub> = v·(ρ/ρ<sub>0</sub>)<sup>1/3</sup>.
- **"All features" is worst:** calendar features let the trees memorise training-period quirks.

---

## D. Forecasting

### D12. Direct multi-horizon models; persistence and linear AR as baselines
- **Direct strategy:** one model per horizon (10 min, 1 h, 3 h, 6 h).
- **Rejected alternative:** a recursive one-step model. It feeds its own errors back in, so errors compound at exactly the long horizons where they hurt.
- **Persistence (P(t+h) = P(t)) is the baseline** every wind forecast must beat. At 10 minutes it is famously hard to beat, because power at lag 1 is 0.98 autocorrelated.
- **Ridge AR(X)** tests whether the problem needs non-linearity at all. It loses to persistence on MAE at every horizon: a linear model cannot represent the power curve's saturation.

### D13. LightGBM predicts the *change* in power, not the level
- **What failed first:** a LightGBM model predicting the level lost to persistence on MAE. With 92% autocorrelation at 1 hour, its splits were spent re-learning "power ≈ current power".
- **The fix:** predict P(t+h) − P(t). Persistence becomes the built-in starting point, and every split goes to what persistence gets wrong.

| Horizon | Persistence nMAE | LightGBM (Δ) nMAE | RMSE skill vs persistence |
|---|---|---|---|
| 10 min | 3.49% | 3.48% | +0.7% |
| 1 h | 7.94% | 7.86% | +2.2% |
| 3 h | 13.01% | 12.79% | +3.7% |
| 6 h | 18.69% | 18.04% | +6.0% |

- **L1 objective:** we report MAE, and L1 also resists the heavy tails created by sudden stops.
- **Honest conclusion:** the gain is real but small, and it grows with horizon.
  - The expanding-window **backtest** at 1 hour shows LightGBM winning in only **5 of 9 months**. At short horizons ML and persistence are effectively tied.
  - The information needed to beat persistence by a lot is not in the turbine's own history. It is in **numerical weather prediction (NWP)** forecasts.
  - SHAP agrees: the model's corrections come mostly from hour-of-day and season, meaning diurnal wind-ramp patterns, not from anything the turbine "knows".
- **Recommendation:** use persistence for ≤1 h. Add an NWP feed before investing in a bigger model for 3–48 h.

### D14. Only rows usable by all models are used for evaluation
- Ridge cannot take NaN, while LightGBM can. All models are scored on the same rows, which costs about 5% of rows next to data gaps.
- **Why:** a comparison on different rows is not a comparison.

---

## E. Anomaly detection

### D15. Two tiers: physical rule for stops, statistical test for subtle loss
- **Tier 1, stops:** wind above 3.5 m/s and no power. A rule finds these with certainty, and no ML is needed.
- **Tier 2, subtle loss:** this is where the value is. Point by point it is invisible: the residual noise is 85–136 kW in the 7–11 m/s bins, the same size as a 5% loss.
- **Method:** NBM residual → standardised per 1 m/s wind bin → one-sided **CUSUM**.
  - **Why per-bin standardisation:** residual noise is tiny near cut-in and at rated power, and largest on the steep part of the curve. One global sigma would over-alarm mid-curve and be blind elsewhere.
  - **Why the standardiser is fitted on validation:** training residuals are optimistic, because the model has already seen those points.
  - **Why CUSUM:** it is the classical optimal test for a *small, persistent* mean shift, which is exactly what degradation looks like. It accumulates evidence and ignores zero-mean noise.
  - **z is clipped at ±4,** so a single glitch can never alarm alone.
- **Threshold `h` is a business decision, not a magic number.** It is set to the smallest value giving **≤1 false alarm per turbine per 30 days** on healthy validation data, which gives h = 194.
  - It is calibrated empirically, not from textbook average-run-length tables. Ten-minute residuals are autocorrelated (lag-1 0.62), which breaks the i.i.d. assumption behind those tables.
- **Reference value `k`** was chosen by injecting faults into **validation** data:

| k | h | 5% loss detected (val) | 10% loss detected (val) |
|---|---|---|---|
| **0.25** | 194 | **68%** | **100%** |
| 0.5 | 140 | 60% | 100% |
| 1.0 | 63 | 28% | 98% |
| 2.0 | 14 | 28% | 85% |

Small k targets small shifts, and small shifts are the ones worth catching.

### D16. Ground truth by fault injection, credited conservatively
- **Why inject:** the dataset has no labelled faults, so I create faults with known start and size.
- **Fault types:**
  - **Efficiency loss** (3, 5, 8, 10%): power is reduced *only in partial load (<12 m/s)*. Physics: above rated, the pitch system recovers the loss. This mimics soiling and pitch offset.
  - **Anemometer drift +5%:** the turbine is healthy and the sensor lies.
- **Protocol:**
  - Each fault persists for 7 days, because a real fault lasts until it is repaired.
  - 40 random start times per scenario.
  - A detection is credited **only if the injected run alarms before any alarm the clean data raises in the same window**. Natural events in December cannot inflate the score.

**Test results** (design frozen before this run):

| Fault | NBM + CUSUM detected | Median delay | Isolation Forest |
|---|---|---|---|
| Efficiency −3% | 42% | 52 h | 0% |
| Efficiency −5% | 78% | 79 h | 0% |
| Efficiency −8% | 85% | 52 h | 0% |
| Efficiency −10% | 90% | 52 h | 0% |
| Anemometer +5% | 98% | 36 h | 55% |

### D17. Isolation Forest as the physics-free comparator
- **Why it is the right challenger:** it is the go-to unsupervised detector and needs no power-curve model.
- **Fairness:** it was given the same persistence logic and the same false-alarm budget, calibrated on the same validation data.
- **Why it fails on efficiency loss:** a 5% loss keeps every point *inside* the normal (wind, power) cloud. Isolation Forest scores points one at a time and has no notion of "systematically a bit low".
- **Why it partly catches anemometer drift:** drift pushes points sideways *out of* the cloud.
- **Takeaway:** domain structure (expected power) beats a generic detector for degradation.

### D18. Rejected: rolling 14-day bias correction
- **The idea:** subtract a trailing 14-day median of z, lagged 1 day, to follow seasonal air-density drift.
- **Result on validation:** 5% detection fell from 68% to 40%. It absorbs part of a real fault within days.
- **Rejected.** Seasonal drift should be fixed with temperature data (D11), not by a detector that learns to forgive.

### D19. Natural events: early warning before stops
- **What the clean test data shows:** CUSUM raised 6 alarm events in Nov–Dec on clean test data. I inspected the 12 hours before each one.
  - **5 of 6 are clear underperformance:** actual/expected energy of 0.40–0.89.
  - **19 November is different:** a 3% shortfall at 15 m/s, consistent with the ~3,450 kW derating setpoint. The deviation is real, but it is probably an *intended* curtailment rather than a fault. With operator status codes it would be suppressed. Without them, the very tight residual scale at rated power (≈10 kW) makes the detector sensitive to setpoint changes.
- **Three episodes (30 Nov, 19 Dec, 25 Dec)** show power sliding well below expected for hours, then the turbine stopping.
  - The detector alarmed **7–16 hours before the first stop** in each episode.
  - The pattern (gradual loss, then shutdown, in winter) is consistent with **blade icing**. Without temperature data this is a hypothesis, not a diagnosis.
- **6 December** shows a cap at about 3,450 kW: a derating or curtailment setpoint, worth confirming with the operator.

---

## F. Business impact (test period, Nov–Dec, one turbine)

| Item | Value |
|---|---|
| Energy lost in stops | 74.1 MWh (~$5,400 at 7.3 USc/kWh) |
| Energy lost while running abnormally | 92.7 MWh (~$6,800) |
| Cost of an undetected 5% partial-load loss | ~$2,040 per month per turbine |
| Naive 90% rule false flags on healthy data | 34% with manufacturer curve vs 8.5% with NBM |
| Forecast error avoided vs persistence (6 h) | 30.9 MWh of absolute error over 2 months |

- **Tariff:** 7.3 US cents/kWh, the Turkish YEKDEM feed-in tariff for wind. This is an assumption for the turbine's actual contract.
- **Scale:** a 5% loss caught in about 3 days instead of at a monthly performance review saves about 90% of that month's loss. Across a 50-turbine farm, one such fault per turbine per year is worth about $90k.

---

## G. Assumptions and how they were checked

| Assumption | Check |
|---|---|
| Training months represent healthy operation | Filter (D7) removes stops and derates; monthly residual bias inspected |
| Power is monotone in wind below cut-out | True in data; enforced by constraint (D9) |
| Residuals are stationary within a season | **Violated across seasons:** mean residual (actual − expected, partial load) swings from −32 kW (July) to +88 kW (February). This is the main limitation |
| Residual noise is independent (CUSUM) | **Violated** (autocorrelation 0.62). Handled by empirical calibration of h |
| Injected faults resemble real ones | Partially. Real faults may ramp rather than step; a step is the easier case for CUSUM |

## H. Where it fails in production, and how I would detect it
1. **Seasonal drift → missed faults in winter, false alarms in summer.**
   - *Detect:* monthly bias of NBM residual on filter-normal data. Alert if |bias| > 1.5% of rated.
   - *Fix:* temperature-based density correction, and retrain quarterly.
2. **Anemometer drift is indistinguishable from real loss.** The NBM trusts the wind sensor.
   - *Detect:* compare against a neighbouring turbine's wind or a met mast; compare nacelle wind against power at rated.
   - *Fix:* route alarms to "check sensor" first when a neighbour disagrees.
3. **Slow degradation over months is absorbed** if the NBM is retrained on degraded data.
   - *Fix:* retrain only on periods signed off as healthy; keep a frozen reference model.
4. **Large, fast events take about 8 h to alarm,** because CUSUM is tuned for small shifts and the per-point contribution is clipped.
   - *Fix:* add a Shewhart-style fast lane (for example, actual < 50% of expected for 1 h → immediate alarm).
5. **Concept drift in the forecaster.**
   - *Detect:* rolling 7-day skill vs persistence. If it drops below 0, fall back to persistence automatically.
