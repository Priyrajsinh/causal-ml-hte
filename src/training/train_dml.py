"""Day 3 orchestrator: fit DML on NSW and CPS-observational, persist artefacts.

Wires the headline three-way ATE comparison (OLS-CPS biased | DML-CPS unbiased
| DML-NSW ≈ RCT truth) and enforces rule C37: the NSW DML 95% CI must contain
the RCT ground-truth ATE (±tolerance from config).
"""

import argparse
import json
import os
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from src.baseline.ate_table import plot_three_way_table
from src.config import load_config
from src.data.skew_check import save_training_stats
from src.data.validation import validate_lalonde_df
from src.evaluation.cate_plots import plot_cate_histogram, plot_cate_with_ci_sorted
from src.logger import get_logger
from src.models.causal_forest_model import CausalForestEstimator
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


def _fit_causal_forest(
    cfg: dict, df: pd.DataFrame
) -> tuple[CausalForestEstimator, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Fit CausalForestDML on df, return (est, cate, ci_lo, ci_hi, ate_dict)."""
    Y, T, X = _split_y_t_x(df)
    n_boot_cfg = int(cfg["causal_forest"]["n_bootstrap_samples"])
    n_boot = 100 if os.getenv("CI", "").lower() == "true" else n_boot_cfg
    cf = CausalForestEstimator(
        n_estimators=cfg["causal_forest"]["n_estimators"],
        min_samples_leaf=cfg["causal_forest"]["min_samples_leaf"],
        max_depth=cfg["causal_forest"]["max_depth"],
        cv=cfg["causal_forest"]["cv"],
        n_bootstrap_samples=n_boot,
        model_y_kwargs={**cfg["dml"]["model_y"], "random_state": cfg["seed"]},
        model_t_kwargs={**cfg["dml"]["model_t"], "random_state": cfg["seed"]},
        seed=cfg["seed"],
    ).fit(Y=Y, T=T, X=X)
    cate, lo, hi = cf.predict_with_ci(X)
    return cf, cate, lo, hi, cf.ate(X)


def main(config_path: str) -> None:
    """Fit DML on NSW + CPS-obs, persist artefacts, assert rule C37."""
    cfg = load_config(config_path)
    nsw = validate_lalonde_df(pd.read_csv(cfg["data"]["nsw_processed"]))
    obs_path = (
        Path(cfg["data"]["nsw_processed"]).parent / "nsw_treated_plus_cps_controls.csv"
    )
    obs = validate_lalonde_df(pd.read_csv(obs_path))

    if int(cfg["dml"]["cv"]) != 5 or int(cfg["causal_forest"]["cv"]) != 5:
        logger.warning("cv != 5 for DML/CF — production runs must use cv=5 (rule C35).")

    dml_nsw, ate_nsw = _fit_dml(cfg, nsw)
    dml_obs, ate_obs = _fit_dml(cfg, obs)

    cf, cate, cate_lo, cate_hi, cf_ate = _fit_causal_forest(cfg, nsw)

    models_dir = Path(cfg["paths"]["models_dir"])
    dml_nsw.save(models_dir / "dml_nsw")
    dml_obs.save(models_dir / "dml_cps")
    cf.save(models_dir / "causal_forest")
    save_training_stats(
        pd.DataFrame(nsw[COVARIATES].to_numpy(), columns=COVARIATES),
        models_dir / "training_stats.json",
    )

    _, _, X_nsw = _split_y_t_x(nsw)
    cate_df = pd.DataFrame(
        {
            "cate": cate,
            "ci_lower": cate_lo,
            "ci_upper": cate_hi,
            **{c: X_nsw[:, i] for i, c in enumerate(COVARIATES)},
        }
    )
    reports_dir = Path(cfg["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    cate_df.to_parquet(reports_dir / "cate_per_row.parquet", index=False)

    cate_distribution = {
        "mean": float(cate.mean()),
        "std": float(cate.std()),
        "min": float(cate.min()),
        "max": float(cate.max()),
        "p10": float(np.percentile(cate, 10)),
        "p50": float(np.percentile(cate, 50)),
        "p90": float(np.percentile(cate, 90)),
        "ate_from_cf": float(cf_ate["ate"]),
        "ate_ci_from_cf": [float(cf_ate["ci_lower"]), float(cf_ate["ci_upper"])],
    }
    (models_dir / "cate_distribution.json").write_text(
        json.dumps(cate_distribution, indent=2)
    )

    figures_dir = Path(cfg["paths"]["figures_dir"])
    hist_path = figures_dir / "cate_histogram.png"
    cater_path = figures_dir / "cate_caterpillar.png"
    plot_cate_histogram(cate, float(cf_ate["ate"]), hist_path)
    plot_cate_with_ci_sorted(cate, cate_lo, cate_hi, cater_path)

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
    with mlflow.start_run(run_name="day3_4_dml_and_forest"):
        mlflow.log_params(
            {
                "cv": cfg["dml"]["cv"],
                "estimator": "LinearDML+CausalForestDML",
                **{f"model_y_{k}": v for k, v in cfg["dml"]["model_y"].items()},
                **{f"model_t_{k}": v for k, v in cfg["dml"]["model_t"].items()},
                "cf_n_estimators": cfg["causal_forest"]["n_estimators"],
                "cf_min_samples_leaf": cfg["causal_forest"]["min_samples_leaf"],
                "cf_max_depth": cfg["causal_forest"]["max_depth"],
                "cf_n_bootstrap_samples": cf.n_bootstrap_samples,
            }
        )
        mlflow.log_metric("dml_nsw_ate", ate_nsw["ate"])
        mlflow.log_metric("dml_nsw_ci_lower", ate_nsw["ci_lower"])
        mlflow.log_metric("dml_nsw_ci_upper", ate_nsw["ci_upper"])
        mlflow.log_metric("dml_cps_ate", ate_obs["ate"])
        mlflow.log_metric("dml_cps_ci_lower", ate_obs["ci_lower"])
        mlflow.log_metric("dml_cps_ci_upper", ate_obs["ci_upper"])
        mlflow.log_metric("cf_ate", float(cf_ate["ate"]))
        mlflow.log_metric("cf_ate_ci_lower", float(cf_ate["ci_lower"]))
        mlflow.log_metric("cf_ate_ci_upper", float(cf_ate["ci_upper"]))
        mlflow.log_metric("cate_mean", float(cate.mean()))
        mlflow.log_metric("cate_std", float(cate.std()))
        mlflow.log_artifact(str(hist_path))
        mlflow.log_artifact(str(cater_path))
        mlflow.log_artifact(str(reports_dir / "cate_per_row.parquet"))
        mlflow.log_artifact(str(models_dir / "cate_distribution.json"))

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
    logger.info(
        "CF ATE:       $%.0f (95%% CI [$%.0f, $%.0f])  cate_std=$%.0f",
        cf_ate["ate"],
        cf_ate["ci_lower"],
        cf_ate["ci_upper"],
        float(cate.std()),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    main(parser.parse_args().config)
