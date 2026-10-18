"""Thin MLflow wrapper so every model comparison is logged the same way."""
from __future__ import annotations

import logging
import warnings

import mlflow

from . import config as C

logging.getLogger("mlflow").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", module="mlflow")
_ready = False


def _init():
    global _ready
    if not _ready:
        mlflow.set_tracking_uri(C.MLFLOW_URI)
        _ready = True


def log_run(experiment: str, run_name: str, params: dict, metrics: dict, tags: dict | None = None):
    _init()
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({k: str(v)[:250] for k, v in params.items()})
        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if v is not None})
        if tags:
            mlflow.set_tags(tags)


def export_runs(path):
    """Write every run of every experiment to one CSV (the 'experiment log')."""
    _init()
    exps = [e.experiment_id for e in mlflow.search_experiments() if e.name != "Default"]
    runs = mlflow.search_runs(experiment_ids=exps)
    names = {e.experiment_id: e.name for e in mlflow.search_experiments()}
    runs["experiment"] = runs["experiment_id"].map(names)
    keep = ["experiment", "tags.mlflow.runName"] + sorted(c for c in runs.columns if c.startswith(("params.", "metrics.")))
    runs[keep].sort_values(["experiment", "tags.mlflow.runName"]).to_csv(path, index=False)
    return runs
