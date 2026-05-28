"""Unit tests for src/evaluation/shap_moderators.py (Day 6 SHAP moderators)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import shap

from src.evaluation.shap_moderators import (
    build_cate_explainer,
    explain_cate,
    plot_shap_bar,
    plot_shap_beeswarm,
    plot_shap_waterfall_individual,
    save_moderator_summary,
)
from src.models.causal_forest_model import CausalForestEstimator
from src.models.dml_model import COVARIATES


def _fast_base() -> dict[str, Any]:
    """Tiny LightGBM kwargs to keep these tests sub-second."""
    return {
        "n_estimators": 20,
        "max_depth": 3,
        "min_child_samples": 1,
        "learning_rate": 0.1,
        "verbose": -1,
        "random_state": 42,
    }


def _fitted_cf(n: int = 60, seed: int = 0) -> tuple[CausalForestEstimator, np.ndarray]:
    """Fit a tiny CausalForestEstimator on synthetic LaLonde-shaped data."""
    rng = np.random.default_rng(seed)
    treat = rng.integers(0, 2, size=n).astype(int)
    re74 = rng.uniform(0, 5000, size=n)
    re75 = rng.uniform(0, 5000, size=n)
    re78 = 1500.0 * treat + 0.4 * re74 + 0.4 * re75 + rng.normal(0, 500, size=n)
    df = pd.DataFrame(
        {
            "age": rng.integers(18, 50, size=n).astype(int),
            "education": rng.integers(7, 16, size=n).astype(int),
            "black": rng.integers(0, 2, size=n).astype(int),
            "hispanic": rng.integers(0, 2, size=n).astype(int),
            "married": rng.integers(0, 2, size=n).astype(int),
            "nodegree": rng.integers(0, 2, size=n).astype(int),
            "re74": re74,
            "re75": re75,
        }
    )
    X = df[COVARIATES].to_numpy(dtype=float)
    cf = CausalForestEstimator(
        n_estimators=20,
        min_samples_leaf=2,
        max_depth=3,
        cv=2,
        n_bootstrap_samples=5,
        model_y_kwargs=_fast_base(),
        model_t_kwargs=_fast_base(),
        seed=42,
    ).fit(Y=re78, T=treat, X=X)
    return cf, X


def test_build_cate_explainer_returns_kernel_explainer() -> None:
    """``build_cate_explainer`` must return a ``shap.KernelExplainer``."""
    cf, X = _fitted_cf()
    explainer = build_cate_explainer(cf.estimator, X, background_size=10)
    assert isinstance(explainer, shap.KernelExplainer)


def test_explain_cate_returns_shap_explanation_with_right_shape() -> None:
    """``explain_cate`` returns an Explanation with shape (n_explain, 8)."""
    cf, X = _fitted_cf()
    expl = explain_cate(cf.estimator, X, n_samples=20, n_explain=10)
    assert isinstance(expl, shap.Explanation)
    assert expl.values.shape == (10, len(COVARIATES))
    assert expl.base_values.shape == (10,)
    assert list(expl.feature_names) == COVARIATES


def test_moderator_summary_has_all_8_covariates_sorted_desc(tmp_path: Path) -> None:
    """All 8 covariates present in summary, values sorted descending."""
    cf, X = _fitted_cf()
    expl = explain_cate(cf.estimator, X, n_samples=20, n_explain=10)
    out = tmp_path / "shap_moderators.json"
    summary = save_moderator_summary(expl, out)
    assert set(summary.keys()) == set(COVARIATES)
    vals = list(summary.values())
    assert vals == sorted(vals, reverse=True)
    assert out.exists()


def test_beeswarm_and_bar_plots_write_files(tmp_path: Path) -> None:
    """The beeswarm + bar plotters produce non-empty PNGs."""
    cf, X = _fitted_cf()
    expl = explain_cate(cf.estimator, X, n_samples=20, n_explain=10)
    beeswarm = tmp_path / "beeswarm.png"
    bar = tmp_path / "bar.png"
    plot_shap_beeswarm(expl, beeswarm)
    plot_shap_bar(expl, bar)
    assert beeswarm.exists() and beeswarm.stat().st_size > 0
    assert bar.exists() and bar.stat().st_size > 0


def test_waterfall_writes_file_for_valid_idx(tmp_path: Path) -> None:
    """The individual waterfall plotter writes a non-empty PNG."""
    cf, X = _fitted_cf()
    expl = explain_cate(cf.estimator, X, n_samples=20, n_explain=10)
    out = tmp_path / "waterfall.png"
    plot_shap_waterfall_individual(expl, idx=0, out_path=out)
    assert out.exists() and out.stat().st_size > 0


def test_explain_cate_subsamples_when_n_explain_lt_n() -> None:
    """When n_explain < |X|, only n_explain rows are explained."""
    cf, X = _fitted_cf(n=60)
    expl = explain_cate(cf.estimator, X, n_samples=20, n_explain=8)
    assert expl.values.shape[0] == 8


def test_explain_cate_uses_full_X_when_n_explain_none() -> None:
    """With n_explain=None, all rows of X get a SHAP explanation."""
    cf, X = _fitted_cf(n=20)
    expl = explain_cate(cf.estimator, X, n_samples=20, n_explain=None)
    assert expl.values.shape[0] == 20


@pytest.mark.slow
def test_full_pipeline_on_real_cate_parquet(tmp_path: Path) -> None:
    """End-to-end sanity using the real reports/cate_per_row.parquet (opt-in)."""
    parquet = Path("reports/cate_per_row.parquet")
    if not parquet.exists():
        pytest.skip("cate_per_row.parquet not materialised (Day 5 not run)")
    cate_df = pd.read_parquet(parquet)
    X = cate_df[COVARIATES].to_numpy(dtype=float)
    cf = CausalForestEstimator.load(Path("models/causal_forest"))
    expl = explain_cate(cf.estimator, X, n_samples=30, n_explain=20)
    assert expl.values.shape == (20, len(COVARIATES))
