"""Unit tests for CausalDMLEstimator (cv guard, save/load, ate, cate intervals)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.exceptions import PredictionError
from src.models.dml_model import COVARIATES, CausalDMLEstimator


def _arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Materialise (Y, T, X) from a LaLonde-shaped DataFrame for unit tests."""
    return (
        df["re78"].to_numpy(dtype=float),
        df["treat"].to_numpy(dtype=int),
        df[COVARIATES].to_numpy(dtype=float),
    )


def _fast_base() -> dict[str, Any]:
    """Lightweight LightGBM kwargs that keep unit tests sub-second."""
    return {
        "n_estimators": 20,
        "max_depth": 3,
        "min_child_samples": 1,
        "learning_rate": 0.1,
        "verbose": -1,
        "random_state": 42,
    }


def _fast_est(cv: int = 2) -> CausalDMLEstimator:
    """Build a tiny CausalDMLEstimator for unit-test speed."""
    return CausalDMLEstimator(
        cv=cv, model_y_kwargs=_fast_base(), model_t_kwargs=_fast_base()
    )


def test_dml_requires_cv_ge_2() -> None:
    """Rule C35: cv < 2 disables cross-fitting and must raise."""
    with pytest.raises(ValueError, match="cv must be >= 2"):
        CausalDMLEstimator(cv=1)


def test_predict_before_fit_raises() -> None:
    """predict before fit must raise PredictionError, not return None."""
    est = _fast_est(cv=2)
    with pytest.raises(PredictionError):
        est.predict(np.zeros((1, len(COVARIATES))))


def test_save_load_roundtrip(dummy_lalonde_df: pd.DataFrame, tmp_path: Path) -> None:
    """save → load round-trip must produce identical CATE predictions."""
    Y, T, X = _arrays(dummy_lalonde_df)
    est = _fast_est(cv=2).fit(Y=Y, T=T, X=X)
    est.save(tmp_path / "dml")
    loaded = CausalDMLEstimator.load(tmp_path / "dml")
    np.testing.assert_allclose(est.predict(X), loaded.predict(X))


def test_ate_returns_finite_ci(dummy_lalonde_df: pd.DataFrame) -> None:
    """ate() must return finite ate, ci_lower, ci_upper on dummy data."""
    Y, T, X = _arrays(dummy_lalonde_df)
    est = _fast_est(cv=2).fit(Y=Y, T=T, X=X)
    result = est.ate(X)
    assert set(result) >= {"ate", "ci_lower", "ci_upper", "alpha"}
    assert np.isfinite(result["ate"])
    assert np.isfinite(result["ci_lower"])
    assert np.isfinite(result["ci_upper"])
    assert result["ci_lower"] <= result["ci_upper"]


def test_cate_intervals_shape(dummy_lalonde_df: pd.DataFrame) -> None:
    """cate_intervals returns two 1-D vectors aligned with rows of X."""
    Y, T, X = _arrays(dummy_lalonde_df)
    est = _fast_est(cv=2).fit(Y=Y, T=T, X=X)
    lo, hi = est.cate_intervals(X)
    assert lo.shape == (X.shape[0],)
    assert hi.shape == (X.shape[0],)
    assert np.all(lo <= hi)
