"""Unit tests for src/heterogeneity analysis and policy modules."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.heterogeneity.analysis import (
    cate_by_category,
    cate_by_quartile,
    moderator_scores,
    plot_cate_by_quartile,
    plot_cate_scatter,
)
from src.heterogeneity.policy import plot_policy_curve, top_fraction_targeting


@pytest.fixture
def cate_df() -> pd.DataFrame:
    """Synthetic CATE DataFrame with clear quartile signal."""
    rng = np.random.default_rng(0)
    n = 80
    education = rng.integers(8, 17, size=n)
    age = rng.integers(18, 55, size=n)
    re74 = rng.uniform(0, 10000, size=n)
    black = rng.integers(0, 2, size=n)
    hispanic = rng.integers(0, 2, size=n)
    married = rng.integers(0, 2, size=n)
    nodegree = rng.integers(0, 2, size=n)
    re75 = rng.uniform(0, 10000, size=n)
    # CATE positively correlated with education so Q4 > Q1
    cate = 200 * education + rng.normal(0, 100, size=n)
    return pd.DataFrame(
        {
            "age": age,
            "education": education,
            "black": black,
            "hispanic": hispanic,
            "married": married,
            "nodegree": nodegree,
            "re74": re74,
            "re75": re75,
            "cate": cate,
            "ci_lower": cate - 200,
            "ci_upper": cate + 200,
        }
    )


def test_cate_by_quartile_returns_4_rows(cate_df: pd.DataFrame) -> None:
    """Quartile grouping produces exactly 4 rows."""
    result = cate_by_quartile(cate_df, "education")
    assert len(result) == 4, f"Expected 4 rows, got {len(result)}"


def test_cate_by_category_groups_correctly(cate_df: pd.DataFrame) -> None:
    """Binary covariate grouping produces exactly 2 rows."""
    result = cate_by_category(cate_df, "black")
    assert len(result) == 2


def test_moderator_scores_sorted_descending(cate_df: pd.DataFrame) -> None:
    """cate_range column is non-increasing (descending sort)."""
    covariates = ["age", "education", "black", "hispanic", "re74"]
    scores = moderator_scores(cate_df, covariates)
    ranges = scores["cate_range"].tolist()
    assert ranges == sorted(ranges, reverse=True)


def test_top_fraction_targeting_returns_correct_size(cate_df: pd.DataFrame) -> None:
    """n_targeted matches round(n * fraction)."""
    result = top_fraction_targeting(cate_df, fraction=0.30)
    expected_k = max(1, round(len(cate_df) * 0.30))
    assert result["n_targeted"] == expected_k


def test_policy_lift_positive_when_signal_present(cate_df: pd.DataFrame) -> None:
    """Targeted group CATE > untargeted CATE when clear ordering exists."""
    result = top_fraction_targeting(cate_df, fraction=0.30)
    top = float(result["mean_cate_targeted"])  # type: ignore[arg-type]
    bot = float(result["mean_cate_untargeted"])  # type: ignore[arg-type]
    assert top > bot


def test_policy_curve_plot_runs(cate_df: pd.DataFrame, tmp_path: Path) -> None:
    """plot_policy_curve produces the output file."""
    out = tmp_path / "policy_curve.png"
    plot_policy_curve(cate_df, out)
    assert out.exists()


def test_cate_quartile_plot_runs(cate_df: pd.DataFrame, tmp_path: Path) -> None:
    """plot_cate_by_quartile produces the output file."""
    summary = cate_by_quartile(cate_df, "education")
    out = tmp_path / "het_edu.png"
    plot_cate_by_quartile(summary, "education", out)
    assert out.exists()


def test_cate_scatter_plot_runs(cate_df: pd.DataFrame, tmp_path: Path) -> None:
    """plot_cate_scatter produces the output file."""
    out = tmp_path / "scatter_edu.png"
    plot_cate_scatter(cate_df, "education", out)
    assert out.exists()


def test_results_json_has_heterogeneity_and_policy_keys(
    cate_df: pd.DataFrame, tmp_path: Path
) -> None:
    """After writing results, both top-level keys are present."""
    results_path = tmp_path / "results.json"
    existing: dict[str, object] = {}

    from src.heterogeneity.analysis import moderator_scores as _ms

    covariates = [
        "age",
        "education",
        "black",
        "hispanic",
        "married",
        "nodegree",
        "re74",
        "re75",
    ]
    edu_q = cate_by_quartile(cate_df, "education")
    age_q = cate_by_quartile(cate_df, "age")
    re74_q = cate_by_quartile(cate_df, "re74")
    binary_summaries = {
        c: cate_by_category(cate_df, c).to_dict(orient="records")
        for c in ["black", "hispanic", "married", "nodegree"]
    }
    scores = _ms(cate_df, covariates)
    policy = top_fraction_targeting(cate_df, fraction=0.30)

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

    loaded = json.loads(results_path.read_text())
    assert "heterogeneity" in loaded
    assert "policy_targeting" in loaded
    assert "education_quartile" in loaded["heterogeneity"]
    assert "n_targeted" in loaded["policy_targeting"]
