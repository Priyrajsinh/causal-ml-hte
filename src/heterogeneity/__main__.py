"""Heterogeneity analysis orchestrator. Run via: python -m src.heterogeneity."""

import argparse
import json
from pathlib import Path

import mlflow
import pandas as pd

from src.config import load_config
from src.heterogeneity.analysis import (
    cate_by_category,
    cate_by_quartile,
    moderator_scores,
    plot_cate_by_quartile,
    plot_cate_scatter,
)
from src.heterogeneity.policy import (
    plot_policy_curve,
    plot_targeted_profile,
    top_fraction_targeting,
)
from src.logger import get_logger

logger = get_logger(__name__)

_COVARIATES = [
    "age",
    "education",
    "black",
    "hispanic",
    "married",
    "nodegree",
    "re74",
    "re75",
]


def main(config_path: str = "config/config.yaml") -> None:
    """Run heterogeneity analysis and policy targeting; log artefacts to MLflow."""
    cfg = load_config(config_path)
    cate_df = pd.read_parquet(
        Path(cfg["paths"]["reports_dir"]) / "cate_per_row.parquet"
    )

    fig_dir = Path(cfg["paths"]["figures_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Heterogeneity by 3 natural moderators
    edu_q = cate_by_quartile(cate_df, "education")
    age_q = cate_by_quartile(cate_df, "age")
    re74_q = cate_by_quartile(cate_df, "re74")
    plot_cate_by_quartile(edu_q, "education", fig_dir / "heterogeneity_education.png")
    plot_cate_by_quartile(age_q, "age", fig_dir / "heterogeneity_age.png")
    plot_cate_by_quartile(re74_q, "re74", fig_dir / "heterogeneity_re74.png")
    plot_cate_scatter(
        cate_df, "education", fig_dir / "heterogeneity_education_scatter.png"
    )
    plot_cate_scatter(cate_df, "age", fig_dir / "heterogeneity_age_scatter.png")
    plot_cate_scatter(cate_df, "re74", fig_dir / "heterogeneity_re74_scatter.png")

    # Binary moderators
    binary_summaries = {
        c: cate_by_category(cate_df, c).to_dict(orient="records")
        for c in ["black", "hispanic", "married", "nodegree"]
    }

    # Moderator score table
    scores = moderator_scores(cate_df, _COVARIATES)
    scores.to_csv(fig_dir / "moderator_scores.csv", index=False)
    logger.info("Top moderator: %s", scores.iloc[0]["covariate"])

    # Policy targeting
    policy = top_fraction_targeting(cate_df, fraction=cfg["policy"]["top_fraction"])
    logger.info(
        "Top %.0f%% mean CATE: $%.0f | bottom mean CATE: $%.0f",
        cfg["policy"]["top_fraction"] * 100,
        policy["mean_cate_targeted"],
        policy["mean_cate_untargeted"],
    )
    plot_policy_curve(cate_df, fig_dir / "policy_curve.png")
    plot_targeted_profile(
        policy["profile_vs_population"],  # type: ignore[arg-type]
        fig_dir / "policy_top30_profile.png",
    )

    # Persist to reports/results.json
    results_path = Path(cfg["paths"]["results_json"])
    existing: dict[str, object] = (
        json.loads(results_path.read_text()) if results_path.exists() else {}
    )
    existing["heterogeneity"] = {
        "education_quartile": edu_q.to_dict(orient="records"),
        "age_quartile": age_q.to_dict(orient="records"),
        "re74_quartile": re74_q.to_dict(orient="records"),
        "binary_groups": binary_summaries,
        "moderator_scores": scores.to_dict(orient="records"),
    }
    existing["policy_targeting"] = policy
    with open(str(results_path), "w") as fh:
        json.dump(existing, fh, indent=2)
    logger.info("results.json updated with heterogeneity + policy_targeting keys")

    # MLflow logging
    top_cate: float = float(policy["mean_cate_targeted"])  # type: ignore[arg-type]
    bot_cate: float = float(policy["mean_cate_untargeted"])  # type: ignore[arg-type]
    total_lift: float = float(policy["estimated_total_lift_usd"])  # type: ignore[arg-type]  # noqa: E501
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])
    with mlflow.start_run(run_name="day5_heterogeneity"):
        mlflow.log_metric("mean_cate_top30", top_cate)
        mlflow.log_metric("mean_cate_bottom70", bot_cate)
        mlflow.log_metric("estimated_total_lift_usd", total_lift)
        for f in fig_dir.glob("heterogeneity_*.png"):
            mlflow.log_artifact(str(f))
        mlflow.log_artifact(str(fig_dir / "policy_curve.png"))
        mlflow.log_artifact(str(fig_dir / "policy_top30_profile.png"))
        mlflow.log_artifact(str(fig_dir / "moderator_scores.csv"))
    logger.info("MLflow day5_heterogeneity run complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Day 5 heterogeneity analysis")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    main(args.config)
