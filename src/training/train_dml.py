"""Day 3 orchestrator: fit DML on NSW and CPS-observational, persist artefacts.

Wires the headline three-way ATE comparison (OLS-CPS biased | DML-CPS unbiased
| DML-NSW ≈ RCT truth) and enforces rule C37: the NSW DML 95% CI must contain
the RCT ground-truth ATE (±tolerance from config).
"""

import argparse
import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from src.baseline.ate_table import plot_three_way_table
from src.config import load_config
from src.data.skew_check import save_training_stats
from src.logger import get_logger
from src.models.dml_model import COVARIATES, CausalDMLEstimator

logger = get_logger(__name__)


def _split_y_t_x(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split a LaLonde DataFrame into (Y, T, X) numpy arrays."""
    return (
        df["re78"].to_numpy(dtype=float),
        df["treat"].to_numpy(dtype=int),
        df[COVARIATES].to_numpy(dtype=float),
    )


def _fit_dml(cfg: dict, df: pd.DataFrame) -> tuple[CausalDMLEstimator, dict]:
    """Fit a fresh CausalDMLEstimator on df and return (estimator, ate_dict)."""
    Y, T, X = _split_y_t_x(df)
    est = CausalDMLEstimator(
        cv=cfg["dml"]["cv"],
        model_y_kwargs={**cfg["dml"]["model_y"], "random_state": cfg["seed"]},
        model_t_kwargs={**cfg["dml"]["model_t"], "random_state": cfg["seed"]},
        seed=cfg["seed"],
    ).fit(Y=Y, T=T, X=X)
    return est, est.ate(X)


def main(config_path: str) -> None:
    """Fit DML on NSW + CPS-obs, persist artefacts, assert rule C37."""
    cfg = load_config(config_path)
    nsw = pd.read_csv(cfg["data"]["nsw_processed"])
    obs_path = (
        Path(cfg["data"]["nsw_processed"]).parent / "nsw_treated_plus_cps_controls.csv"
    )
    obs = pd.read_csv(obs_path)

    dml_nsw, ate_nsw = _fit_dml(cfg, nsw)
    dml_obs, ate_obs = _fit_dml(cfg, obs)

    models_dir = Path(cfg["paths"]["models_dir"])
    dml_nsw.save(models_dir / "dml_nsw")
    dml_obs.save(models_dir / "dml_cps")
    save_training_stats(
        pd.DataFrame(nsw[COVARIATES].to_numpy(), columns=COVARIATES),
        models_dir / "training_stats.json",
    )

    meta = {
        "fitted_on": "nsw_clean.csv",
        "cv": cfg["dml"]["cv"],
        "estimator": "LinearDML",
        "model_y": "LGBMRegressor",
        "model_t": "LGBMClassifier",
        "n_samples_nsw": int(len(nsw)),
        "n_samples_obs": int(len(obs)),
        "seed": cfg["seed"],
    }
    (models_dir / "calibration_metadata.json").write_text(json.dumps(meta, indent=2))

    results_path = Path(cfg["paths"]["results_json"])
    existing = json.loads(results_path.read_text()) if results_path.exists() else {}
    existing["ate_table"] = {
        "ols_cps_unadjusted": existing.get("ols_cps_unadjusted"),
        "ols_cps_adjusted": existing.get("ols_cps_adjusted"),
        "dml_cps": ate_obs,
        "dml_nsw": ate_nsw,
        "rcl_ground_truth": existing.get("rcl_ground_truth"),
    }
    with open(str(results_path), "w") as fh:
        json.dump(existing, fh, indent=2)

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])
    with mlflow.start_run(run_name="day3_dml"):
        mlflow.log_params(
            {
                "cv": cfg["dml"]["cv"],
                "estimator": "LinearDML",
                **{f"model_y_{k}": v for k, v in cfg["dml"]["model_y"].items()},
                **{f"model_t_{k}": v for k, v in cfg["dml"]["model_t"].items()},
            }
        )
        mlflow.log_metric("dml_nsw_ate", ate_nsw["ate"])
        mlflow.log_metric("dml_nsw_ci_lower", ate_nsw["ci_lower"])
        mlflow.log_metric("dml_nsw_ci_upper", ate_nsw["ci_upper"])
        mlflow.log_metric("dml_cps_ate", ate_obs["ate"])
        mlflow.log_metric("dml_cps_ci_lower", ate_obs["ci_lower"])
        mlflow.log_metric("dml_cps_ci_upper", ate_obs["ci_upper"])

    plot_three_way_table(
        results_path,
        Path(cfg["paths"]["figures_dir"]) / "ate_three_way_table.png",
    )

    truth = float(existing["rcl_ground_truth"]["ate"])
    tol = float(cfg["ground_truth"]["tolerance"])
    assert ate_nsw["ci_lower"] - tol <= truth <= ate_nsw["ci_upper"] + tol, (
        f"DML-NSW 95% CI [{ate_nsw['ci_lower']:.0f}, {ate_nsw['ci_upper']:.0f}] "
        f"does not contain RCT truth ${truth:.0f} ± ${tol:.0f} (rule C37). "
        "Inspect nuisance R² in MLflow — likely under-fit or over-fit."
    )

    logger.info(
        "DML-NSW ATE: $%.0f (95%% CI [$%.0f, $%.0f])",
        ate_nsw["ate"],
        ate_nsw["ci_lower"],
        ate_nsw["ci_upper"],
    )
    logger.info(
        "DML-CPS ATE: $%.0f (95%% CI [$%.0f, $%.0f])",
        ate_obs["ate"],
        ate_obs["ci_lower"],
        ate_obs["ci_upper"],
    )
    logger.info("RCT truth:    $%.0f", truth)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    main(parser.parse_args().config)
