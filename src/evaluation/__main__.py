"""Day 6 SHAP moderator orchestrator. Run via: python -m src.evaluation."""

import argparse
import json
from pathlib import Path

import mlflow
import pandas as pd

from src.config import load_config
from src.evaluation.shap_moderators import (
    explain_cate,
    plot_shap_bar,
    plot_shap_beeswarm,
    plot_shap_waterfall_individual,
    save_moderator_summary,
)
from src.logger import get_logger
from src.models.causal_forest_model import CausalForestEstimator
from src.models.dml_model import COVARIATES

logger = get_logger(__name__)


def main(config_path: str = "config/config.yaml") -> None:
    """Compute SHAP moderators on the trained causal forest; persist + log."""
    cfg = load_config(config_path)
    cf = CausalForestEstimator.load(Path(cfg["paths"]["models_dir"]) / "causal_forest")
    cate_df = pd.read_parquet(
        Path(cfg["paths"]["reports_dir"]) / "cate_per_row.parquet"
    )
    X = cate_df[COVARIATES].to_numpy(dtype=float)

    expl = explain_cate(cf.estimator, X, n_samples=100, n_explain=80)

    fig_dir = Path(cfg["paths"]["figures_dir"])
    plot_shap_beeswarm(expl, fig_dir / "shap_moderators_beeswarm.png")
    plot_shap_bar(expl, fig_dir / "shap_moderators_bar.png")
    top_idx = int(cate_df["cate"].values.argmax())
    plot_shap_waterfall_individual(
        expl,
        idx=min(top_idx, len(expl.values) - 1),
        out_path=fig_dir / "shap_moderators_waterfall_top.png",
    )

    summary = save_moderator_summary(
        expl, Path(cfg["paths"]["reports_dir"]) / "shap_moderators.json"
    )

    results_path = Path(cfg["paths"]["results_json"])
    existing: dict[str, object] = (
        json.loads(results_path.read_text()) if results_path.exists() else {}
    )
    existing["shap_moderators"] = summary
    with open(str(results_path), "w") as fh:
        json.dump(existing, fh, indent=2)
    logger.info("results.json updated with shap_moderators key")

    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])
    with mlflow.start_run(run_name="day6_shap_moderators"):
        for feat, val in summary.items():
            mlflow.log_metric(f"shap_mod_mean_abs__{feat}", val)
        for f in [
            fig_dir / "shap_moderators_beeswarm.png",
            fig_dir / "shap_moderators_bar.png",
            fig_dir / "shap_moderators_waterfall_top.png",
        ]:
            mlflow.log_artifact(str(f))
    logger.info("MLflow day6_shap_moderators run complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Day 6 SHAP moderator analysis")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    main(args.config)
