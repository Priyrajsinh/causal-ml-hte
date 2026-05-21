"""Covariate balance table + bootstrap RCT ground-truth ATE (rule C37)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.logger import get_logger

logger = get_logger(__name__)

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


def standardised_mean_difference(df: pd.DataFrame, col: str) -> float:
    """SMD for a single covariate between treat==1 and treat==0."""
    t = df.loc[df["treat"] == 1, col]
    c = df.loc[df["treat"] == 0, col]
    pooled_std = float(np.sqrt((t.var(ddof=1) + c.var(ddof=1)) / 2))
    if pooled_std == 0:
        return 0.0
    return float((t.mean() - c.mean()) / pooled_std)


def covariate_balance_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with mean_treat / mean_control / SMD per covariate."""
    rows = []
    for c in COVARIATES:
        rows.append(
            {
                "covariate": c,
                "mean_treat": float(df.loc[df["treat"] == 1, c].mean()),
                "mean_control": float(df.loc[df["treat"] == 0, c].mean()),
                "smd": standardised_mean_difference(df, c),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_ate_ci(
    df: pd.DataFrame, n_boot: int = 1000, seed: int = 42
) -> dict[str, float | int]:
    """Bootstrap a 95% CI for the simple difference-of-means ATE on an RCT."""
    rng = np.random.default_rng(seed)
    ates = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(df), size=len(df))
        sample = df.iloc[idx]
        ate = float(
            sample.loc[sample["treat"] == 1, "re78"].mean()
            - sample.loc[sample["treat"] == 0, "re78"].mean()
        )
        ates.append(ate)
    return {
        "ate": float(
            df.loc[df["treat"] == 1, "re78"].mean()
            - df.loc[df["treat"] == 0, "re78"].mean()
        ),
        "ci_lower": float(np.percentile(ates, 2.5)),
        "ci_upper": float(np.percentile(ates, 97.5)),
        "se": float(np.std(ates)),
        "n_bootstrap": n_boot,
    }


def save_ground_truth(nsw: pd.DataFrame, out_path: Path) -> dict[str, float | int]:
    """Compute + persist the RCT ground-truth ATE to results.json."""
    ate = bootstrap_ate_ci(nsw)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    existing: dict = (
        json.loads(Path(out_path).read_text()) if Path(out_path).exists() else {}
    )
    existing["rcl_ground_truth"] = ate
    with open(str(out_path), "w") as fh:
        json.dump(existing, fh, indent=2)
    logger.info(
        "RCT ground-truth ATE: $%.0f (95%% CI [$%.0f, $%.0f])",
        ate["ate"],
        ate["ci_lower"],
        ate["ci_upper"],
    )
    return ate
