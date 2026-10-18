"""Project-wide constants.

Every number here is a design decision; the reason is next to it so that a
reviewer (or future me) can challenge it without reading the code.
"""
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[2]
RAW_CSV = ROOT / "data" / "raw" / "T1.csv"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
MLFLOW_URI = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"

SEED = 42

# ---------------------------------------------------------------- turbine physics
# Rated power: the manufacturer curve saturates at exactly 3600 kW (EDA).
RATED_KW = 3600.0
# Cut-in: the manufacturer curve first becomes > 0 at 3.0 m/s (EDA).
CUT_IN_MS = 3.0
# A turbine producing ~0 kW above this speed is stopped, not idling.
# 0.5 m/s margin above cut-in avoids labelling start-up hysteresis as a stop.
STOP_WS_MS = 3.5
STOP_POWER_KW = 1.0
# The manufacturer curve is still at rated at 25.2 m/s (max in data), so the
# cut-out is >= 25 m/s; we use 25 m/s as the upper bound of the operating range.
CUT_OUT_MS = 25.0
# Below rated wind speed the pitch system cannot compensate a loss of
# aerodynamic efficiency, so efficiency faults are visible there only.
PARTIAL_LOAD_MAX_MS = 12.0

# ---------------------------------------------------------------- data split
# Strictly chronological. One year of data, so:
#   train      Jan-Aug  (8 months, includes winter AND summer -> both air densities)
#   validation Sep-Oct  (model selection, early stopping, threshold calibration)
#   test       Nov-Dec  (touched once, at the end)
TRAIN_END = "2018-09-01"
VAL_END = "2018-11-01"

# ---------------------------------------------------------------- normal-behaviour filter
BIN_WIDTH_MS = 0.5        # IEC 61400-12-1 method-of-bins width
FILTER_K = 3.0            # keep points within 3 robust sigmas of the bin median
FILTER_MIN_SCALE_KW = 18  # floor = 0.5% of rated; stops near-zero MAD at rated power
                          # from rejecting normal +/-20 kW jitter

# ---------------------------------------------------------------- forecasting
HORIZONS = {"10min": 1, "1h": 6, "3h": 18, "6h": 36}  # steps of 10 minutes

# ---------------------------------------------------------------- anomaly detection
CUSUM_K = 0.25                 # reference value = half of a 0.5-sigma shift we want to catch
FALSE_ALARMS_PER_30D = 1.0     # alarm budget agreed with "operations"
TARIFF_USD_PER_KWH = 0.073     # Turkish YEKDEM feed-in tariff for wind (7.3 USc/kWh)
