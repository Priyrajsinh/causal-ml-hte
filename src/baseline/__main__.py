"""Orchestrator CLI: run via `python -m src.baseline`.

Loads NSW + CPS, builds observational dataset, computes covariate balance,
RCT ground-truth ATE, and OLS baselines. Logs all metrics to MLflow.
"""

from pathlib import Path

import mlflow

from src.baseline.balance import covariate_balance_table, save_ground_truth
from src.baseline.baseline_ols import save_ols_baselines
from src.config import load_config
from src.data.dataset import build_cps_observational, load_cps, load_nsw
from src.logger import get_logger

logger = get_logger(__name__)


def main(config_path: str = "config/config.yaml") -> None:
    """Run Day 2 baseline pipeline: data loading → balance → ATE → OLS."""
    cfg = load_config(config_path)

    nsw = load_nsw(cfg["data"]["nsw_raw"])
    cps = load_cps(cfg["data"]["cps_raw"])
    obs = build_cps_observational(nsw, cps)

    Path(cfg["data"]["nsw_processed"]).parent.mkdir(parents=True, exist_ok=True)
    nsw.to_csv(cfg["data"]["nsw_processed"], index=False)
    cps.to_csv(cfg["data"]["cps_processed"], index=False)
    obs_path = (
        Path(cfg["data"]["nsw_processed"]).parent / "nsw_treated_plus_cps_controls.csv"
    )
    obs.to_csv(obs_path, index=False)

    figures_dir = Path(cfg["paths"]["figures_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)

    balance = covariate_balance_table(nsw)
    balance_csv = figures_dir / "balance_table.csv"
    balance.to_csv(balance_csv, index=False)

    results_path = Path(cfg["paths"]["results_json"])
    rcl_truth = save_ground_truth(nsw, results_path)
    ols = save_ols_baselines(nsw, obs, results_path)

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])
    with mlflow.start_run(run_name="day2_baselines"):
        mlflow.log_metric("rcl_ground_truth_ate", rcl_truth["ate"])
        mlflow.log_metric("rcl_ground_truth_ci_lower", rcl_truth["ci_lower"])
        mlflow.log_metric("rcl_ground_truth_ci_upper", rcl_truth["ci_upper"])
        mlflow.log_metric("ols_nsw_ate", ols["ols_nsw"]["ate"])
        mlflow.log_metric("ols_cps_unadjusted_ate", ols["ols_cps_unadjusted"]["ate"])
        mlflow.log_metric("ols_cps_adjusted_ate", ols["ols_cps_adjusted"]["ate"])
        mlflow.log_artifact(str(balance_csv))

    logger.info("Day 2 baseline pipeline complete.")


if __name__ == "__main__":
    main()
