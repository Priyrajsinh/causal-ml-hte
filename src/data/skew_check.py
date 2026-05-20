"""Training-distribution skew checker for the 8 LaLonde covariates (rule C23)."""

import json
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

COVARIATES = [
    "age",
    "education",
    "black",
    "hispanic",
    "married",
    "nodegree",
    "re74",
    "re75",
]


def save_training_stats(X: Union[pd.DataFrame, np.ndarray], path: Path) -> None:
    """Save per-feature mean/std/min/max for the 8 LaLonde covariates.

    Used by the FastAPI /cate route to flag inputs outside the training
    distribution (rule C23).
    """
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=COVARIATES)
    stats = {
        c: {
            "mean": float(X[c].mean()),
            "std": float(X[c].std()),
            "min": float(X[c].min()),
            "max": float(X[c].max()),
        }
        for c in X.columns
        if c in COVARIATES
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), "w") as fh:
        json.dump(stats, fh, indent=2)


def check_skew(x_row: np.ndarray, stats_path: Path) -> dict[str, bool]:
    """For a single 8-vector, return {feature: out_of_range_bool}."""
    with open(str(stats_path)) as fh:
        stats = json.load(fh)
    out: dict[str, bool] = {}
    for i, c in enumerate(COVARIATES):
        s = stats[c]
        out[c] = bool(x_row[i] < s["min"] or x_row[i] > s["max"])
    return out
