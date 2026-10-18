# 5-minute walkthrough: script

The assignment's structure is 1 min problem, 2 min decisions, 1.5 min results, 0.5 min next. Don't narrate code. Show figures, speak to decisions.
Record with screen + voice (OBS or Zoom local recording). Have these open in tabs: README, `f1`, `f2`, `f3`, `f7`, `f9b`, `f5`.

---

### 0:00–1:00 · Problem and why Track C  *(show README top, then f1_power_curve)*
> "Wind turbines lose revenue in two ways. Stops are easy to see. The expensive, silent one is a turbine that keeps running a few percent below its potential, from blade soiling, pitch misalignment or icing. On this 3.6 MW turbine, a 5% loss is about two thousand dollars a month.
>
> Operators usually compare against the manufacturer's power curve. This plot is the key finding of the EDA: the manufacturer curve, in orange, sits about 10% above reality. Measured against it, a simple 90% rule flags a third of perfectly healthy data, so operators learn to ignore alarms.
>
> So the core problem is: what *should* this turbine produce right now if it were healthy? That's a normal-behaviour model, and I use it for detection and for forecasting.
>
> I chose Track C because this is structured sensor data governed by physics, power proportional to wind speed cubed. The questions are numeric. An LLM adds nothing until the alarm itself is trustworthy."

### 1:00–3:00 · Key decisions and what I ruled out  *(f2, then f3, then DECISIONS.md table D11)*
> "Four decisions matter most.
>
> **One: no shuffling, and all choices on validation.** Neighbouring records are 98% correlated, so a random split would test on near-copies of training rows. Train is January to August, validation September–October, test November–December, touched once.
>
> **Two: the model.** I compared four families: the manufacturer curve, the IEC binned curve that the industry uses, a physics-shaped logistic, and LightGBM. LightGBM wins by 11% over the IEC curve, mostly in the steep 8 to 11 m/s region, shown here. I rejected neural nets: 25 thousand rows, four tabular features. I gave it **no lagged power**, because then it would copy a fault and the residual would vanish. And I added a monotone constraint. It's accuracy-neutral, but it stops the curve from dipping where data is thin, and a dip becomes a false alarm.
>
> **Three, the most interesting result.** *(f2)* Same wind, about 15% more power in winter, because cold air is denser. Day-of-year as a feature hurt on validation but would have helped on test. I didn't switch, because that would be test leakage. The real fix is the missing variable, temperature.
>
> **Four: detection.** Point-by-point you can't see a 5% loss; the noise is the same size. CUSUM accumulates small persistent shifts. Its threshold isn't a magic number. It's set so a healthy turbine raises at most one false alarm a month, and the sensitivity k was chosen by injecting faults into validation data."

### 3:00–4:30 · Results and what didn't work  *(f7, then f9b, then f5)*
> "Results on the test months. *(f7)* With the same false-alarm budget, CUSUM catches 90% of 10% losses and 78% of 5% losses within a week. Isolation Forest catches none. A 5% loss keeps every point inside the normal cloud, and that's why domain structure beats a generic detector.
>
> *(f9b)* On the clean test data the detector found real winter events. Power slides below expected and the turbine then stops. It alarmed 7 to 16 hours before each stop. It's consistent with icing, but without temperature that's a hypothesis.
>
> What didn't work: *(f5)* my first forecaster predicted the power level and lost to persistence. Switching to predicting the *change* fixed it, but the gain is honest and small, 6% at 6 hours. Without weather forecasts, persistence is near the ceiling. A rolling bias correction for seasons also failed: it absorbed the faults themselves, and I rejected it."

### 4:30–5:00 · With more time
> "Three things: add temperature and pressure for an air-density correction; add a fast lane for big sudden losses, because CUSUM takes about 8 hours on those; and add a neighbour-turbine wind check to separate sensor drift from real loss. Then, and only then, put a GenAI agent on top to turn alarms into work orders. Thank you."
