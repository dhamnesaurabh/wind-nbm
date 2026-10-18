"""Build the 2-page write-up PDF (11 pt minimum, as required)."""
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import Image, ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

OUT = Path(__file__).parent / "Saurabh_Dhamne_TrackC_writeup.pdf"
INK, MUTED, RULE = colors.HexColor("#0b0b0b"), colors.HexColor("#52514e"), colors.HexColor("#d9d8d4")
BLUE = colors.HexColor("#1f5fa8")

base = ParagraphStyle("b", fontName="Helvetica", fontSize=11, leading=13.2, textColor=INK, alignment=TA_LEFT)
small = ParagraphStyle("s", parent=base, fontSize=11, leading=13, textColor=MUTED)
h1 = ParagraphStyle("h1", parent=base, fontName="Helvetica-Bold", fontSize=14, leading=17)
h2 = ParagraphStyle("h2", parent=base, fontName="Helvetica-Bold", fontSize=11.5, leading=14, textColor=BLUE, spaceBefore=5, spaceAfter=2)
cell = ParagraphStyle("c", parent=base, fontSize=11, leading=12.4)
cellb = ParagraphStyle("cb", parent=cell, fontName="Helvetica-Bold")

FIG = Path(__file__).resolve().parents[1] / "results" / "figures"


def figure(name, width_cm):
    from PIL import Image as PILImage
    w, h = PILImage.open(FIG / name).size
    return Image(str(FIG / name), width=width_cm * cm, height=width_cm * cm * h / w)


def P(t, s=base):
    return Paragraph(t, s)


def bullets(items):
    return ListFlowable([ListItem(P(t), leftIndent=10, value="•") for t in items],
                        bulletType="bullet", start="•", leftIndent=10, bulletFontSize=9, spaceBefore=0)


def table(rows, widths):
    data = [[P(c, cellb if i == 0 else cell) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, INK), ("LINEBELOW", (0, 1), (-1, -1), 0.3, RULE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 1.2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.2), ("LEFTPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


story = [
    P("Saurabh Dhamne · Track C (Classical ML / Forecasting) · Scenario S4 applied to wind energy", small),
    P("Wind turbine normal-behaviour model, underperformance detection and short-term forecasting", h1),
    Spacer(1, 3),

    P("1. Problem &amp; domain", h2),
    P("A turbine loses revenue silently. Stops are easy to see, but a turbine that keeps running a few percent "
      "below its potential (soiling, pitch offset, icing, sensor drift) never trips an alarm. On this 3.6 MW turbine a 5% loss costs "
      "~$2,000/month at the 7.3 USc/kWh Turkish wind tariff. Operators compare against the manufacturer curve, but it "
      "<b>over-predicts this turbine by ~10%</b>: a 90%-of-expected rule flags 34% of healthy records, so alarms get ignored. "
      "<b>The opportunity:</b> learn what a healthy turbine should produce (a normal-behaviour model, NBM), then use it to "
      "(a) catch subtle losses early and (b) forecast output. "
      "<b>Why Track C:</b> the data is structured 10-min sensor data governed by a physical law (P ∝ ½ρAC<sub>p</sub>v<super>3</super>). "
      "The questions are numeric, and LLMs add nothing to that. A GenAI layer (alarm → work order) belongs on top only once the alarm is reliable. "
      "<b>Data:</b> Kaggle Wind Turbine SCADA, one turbine in Turkey, 2018, 50,530 rows. Real messiness: 3.9% missing, "
      "4.2% stops, a derating plateau, a ~15% seasonal air-density effect. Limits: one turbine-year, and no temperature, status codes or fault labels."),

    P("2. Approach &amp; algorithm decisions (each with the alternative rejected)", h2),
    bullets([
        "<b>Chronological split</b> (train Jan–Aug, validation Sep–Oct, test Nov–Dec). <b>Every choice is made on validation.</b> "
        "Random split rejected: lag-1 autocorrelation is 0.98, so it would test on near-copies of training rows.",
        "<b>Normal-operation filter</b> before training the NBM: physical state rules, then per-0.5 m/s-bin median ± 3·MAD (IEC bins). "
        "Mean/std rejected, because the derates being removed would inflate the std and hide themselves.",
        "<b>NBM = monotone LightGBM on wind speed, turbulence and direction.</b> Compared against the manufacturer curve, the IEC binned curve "
        "and a 4-parameter logistic. Rejected: NN/LSTM (25k rows, 4 tabular features; a sequence input would let past power leak in), "
        "Gaussian process (O(n<super>3</super>); kept as the upgrade if intervals are needed), random forest (no monotone constraints). "
        "<b>No lagged power</b>, or the model copies the fault and the residual vanishes. The monotone constraint is accuracy-neutral "
        "(1.795 vs 1.807% val) but guarantees a physical curve where data is thin, where a dip would become a false alarm.",
        "<b>Feature ablation (validation):</b> direction and turbulence help, hour does not. <b>Day-of-year hurts on validation but would help on test.</b> "
        "Switching would be test leakage, so I did not. The real fix is the missing variable (temperature → air-density correction).",
        "<b>Forecast:</b> direct per-horizon models (recursive rejected: errors compound). A LightGBM predicting the power <i>level</i> lost to "
        "persistence, so it predicts the <b>change</b> P(t+h)−P(t), making persistence its starting point. Ridge AR tested and rejected (cannot saturate at rated power).",
        "<b>Detection:</b> residual → z per 1 m/s bin (noise is heteroscedastic) → one-sided <b>CUSUM</b>, the classical test for a small persistent "
        "shift. Threshold h set for <b>≤1 false alarm/turbine/month</b> on healthy validation data, calibrated empirically because residuals "
        "are autocorrelated. k = 0.25 chosen by injecting faults into validation. Rejected: point thresholds (noise ≈ fault size), "
        "Isolation Forest (tested, below), and a rolling bias correction (tested: absorbs the fault, 5% detection 68% → 40%).",
    ]),

    P("3. Results &amp; error analysis (held-out test, Nov–Dec)", h2),
    table([
        ["NBM (test, normal)", "nMAE", "Forecast, test nMAE", "Persist.", "LightGBM"],
        ["Manufacturer curve", "3.27%", "10 min", "3.49%", "3.48%"],
        ["IEC binned curve", "2.46%", "1 h", "7.94%", "7.86%"],
        ["4-param. logistic", "3.10%", "3 h", "13.01%", "12.79%"],
        ["<b>LightGBM, monotone</b>", "<b>2.20%</b>", "6 h", "18.69%", "18.04%"],
    ], [4.8 * cm, 1.8 * cm, 4.4 * cm, 2.2 * cm, 2.4 * cm]),
    Spacer(1, 3),
    table([
        ["7-day injected fault, 40 trials", "CUSUM", "Median delay", "Iso. Forest"],
        ["Efficiency loss 3% / 5%", "42% / 78%", "52 h / 79 h", "0% / 0%"],
        ["Efficiency loss 10%", "90%", "52 h", "0%"],
        ["Anemometer drift +5%", "98%", "36 h", "55%"],
    ], [6.2 * cm, 2.6 * cm, 3.0 * cm, 2.8 * cm]),
    Spacer(1, 2),
    bullets([
        "Faults are injected only below rated wind (above it, the pitch system recovers the loss). A detection is credited only if it comes "
        "<b>before</b> any alarm on the clean data. Isolation Forest fails because a 5% loss keeps every point inside the normal cloud.",
        "On clean test data, CUSUM flagged <b>three winter episodes 7–16 h before the turbine stopped</b> (consistent with icing; unconfirmed "
        "without temperature). One alarm (19 Nov) was a ~3% shortfall at a derating setpoint: real, but likely intended.",
        "<b>Where it fails:</b> (1) seasonal density: mean residual (actual − expected) swings −32 to +88 kW, masking faults in winter; "
        "(2) anemometer drift looks identical to real loss; (3) large sudden losses take ~8 h, because CUSUM is tuned for small shifts; "
        "(4) forecast gains are small and the 1-h gain wins only 5/9 backtest months. Without NWP weather input, persistence is near the ceiling.",
    ]),

    P("4. Production &amp; limitations", h2),
    P("<b>Production consideration:</b> score each turbine every 10 min as a stateless service (model + per-bin scales + CUSUM state per "
      "turbine), with alarms into the maintenance queue. Monitor two health signals: monthly NBM bias on filter-normal data "
      "(retrain if |bias| &gt; 1.5% of rated), and rolling 7-day forecast skill (fall back to persistence automatically if &lt; 0). "
      "Retrain quarterly, only on periods signed off as healthy, so slow degradation is never learned as normal. "
      "<b>Limitation before real deployment:</b> no air density. Add temperature and pressure (IEC v·(ρ/ρ<sub>0</sub>)<sup>1/3</sup> "
      "correction), and add a neighbour-turbine wind cross-check to separate sensor drift from real loss. "
      "<b>Value:</b> 74 MWh (~$5.4k) lost in stops and 93 MWh (~$6.8k) in abnormal running in two months on one turbine. Catching a 5% loss in "
      "~3 days instead of at a monthly review saves ~90% of it, about $90k/year for a 50-turbine farm with one such fault per turbine per year."),
    Spacer(1, 6),
    Table([[figure("f7_detection_rate.png", 8.9), figure("f9b_event_12-24.png", 8.9)],
           [P("Fig. 1. Same false-alarm budget: physics residual + CUSUM vs Isolation Forest on injected faults.", small),
            P("Fig. 2. Natural event, 25 Dec: power slides below expected; CUSUM alarms ~16 h before the stop.", small)]],
          colWidths=[9.0 * cm, 9.0 * cm], hAlign="LEFT",
          style=[("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0)]),
]

doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=1.5 * cm, rightMargin=1.5 * cm,
                        topMargin=1.2 * cm, bottomMargin=1.1 * cm,
                        title="Track C write-up - Saurabh Dhamne", author="Saurabh Dhamne")
doc.build(story)
print("wrote", OUT)
