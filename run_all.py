"""Reproduce every number, table and figure in the write-up.

    python scripts/download_data.py   # once
    python run_all.py                 # ~5-10 min on a laptop CPU
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
warnings.filterwarnings("ignore")

from windnbm import config as C  # noqa: E402
from windnbm.pipeline import run_all  # noqa: E402
from windnbm.tracking import export_runs  # noqa: E402

if __name__ == "__main__":
    if C.MLFLOW_URI.startswith("sqlite"):
        db = C.ROOT / "mlflow.db"
        if db.exists():
            db.unlink()  # fresh experiment log on every full run
    run_all()
    export_runs(C.RESULTS / "mlflow_runs.csv")
    print("Done. See results/metrics.json, results/figures/, results/mlflow_runs.csv")
