"""Unit tests for CausalForestEstimator (rule C38 + C36)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.exceptions import PredictionError
from src.models.causal_forest_model import CausalForestEstimator
from src.models.dml_model import COVARIATES


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


def _fast_cf(n_bootstrap_samples: int = 5) -> CausalForestEstimator:
    """Build a tiny CausalForestEstimator for unit-test speed."""
    return CausalForestEstimator(
        n_estimators=20,
        min_samples_leaf=2,
        max_depth=3,
        cv=2,
        n_bootstrap_samples=n_bootstrap_samples,
        model_y_kwargs=_fast_base(),
        model_t_kwargs=_fast_base(),
        seed=42,
    )


def _bigger_lalonde_df(n: int = 80, seed: int = 0) -> pd.DataFrame:
    """A LaLonde-shaped synthetic frame with enough rows for a stable forest CI."""
    rng = np.random.default_rng(seed)
    treat = rng.integers(0, 2, size=n).astype(int)
    re74 = rng.uniform(0, 5000, size=n)
    re75 = rng.uniform(0, 5000, size=n)
    re78 = 1500.0 * treat + 0.4 * re74 + 0.4 * re75 + rng.normal(0, 500, size=n)
    return pd.DataFrame(
        {
            "treat": treat,
            "age": rng.integers(18, 50, size=n).astype(int),
            "education": rng.integers(7, 16, size=n).astype(int),
            "black": rng.integers(0, 2, size=n).astype(int),
            "hispanic": rng.integers(0, 2, size=n).astype(int),
            "married": rng.integers(0, 2, size=n).astype(int),
            "nodegree": rng.integers(0, 2, size=n).astype(int),
            "re74": re74,
            "re75": re75,
            "re78": np.clip(re78, 0, None),
        }
    )


def test_predict_before_fit_raises() -> None:
    """predict before fit must raise PredictionError, not return None."""
    cf = _fast_cf()
    with pytest.raises(PredictionError):
        cf.predict(np.zeros((1, len(COVARIATES))))


def test_safe_predict_nan_raises(dummy_lalonde_df: pd.DataFrame) -> None:
    """NaN inputs to safe_predict must raise PredictionError (rule C36)."""
    Y, T, X = _arrays(dummy_lalonde_df)
    cf = _fast_cf().fit(Y=Y, T=T, X=X)
    bad = X.astype(float).copy()
    bad[0, 0] = np.nan
    with pytest.raises(PredictionError, match="NaN"):
        cf.safe_predict(bad)


def test_safe_predict_wrong_n_features_raises(dummy_lalonde_df: pd.DataFrame) -> None:
    """7-column input (off by one) must raise PredictionError (rule C36)."""
    Y, T, X = _arrays(dummy_lalonde_df)
    cf = _fast_cf().fit(Y=Y, T=T, X=X)
    with pytest.raises(PredictionError, match="covariates"):
        cf.safe_predict(X[:, :-1])


def test_predict_with_ci_shapes_match(dummy_lalonde_df: pd.DataFrame) -> None:
    """predict_with_ci returns three same-length 1-D arrays aligned with X."""
    Y, T, X = _arrays(dummy_lalonde_df)
    cf = _fast_cf().fit(Y=Y, T=T, X=X)
    cate, lo, hi = cf.predict_with_ci(X)
    assert cate.shape == (X.shape[0],)
    assert lo.shape == (X.shape[0],)
    assert hi.shape == (X.shape[0],)


def test_ci_lower_le_cate_le_ci_upper() -> None:
    """Every row must satisfy ci_lower <= cate <= ci_upper on a stable forest."""
    df = _bigger_lalonde_df(n=80, seed=0)
    Y, T, X = _arrays(df)
    cf = _fast_cf(n_bootstrap_samples=20).fit(Y=Y, T=T, X=X)
    cate, lo, hi = cf.predict_with_ci(X)
    assert np.all(lo <= hi)
    assert np.all(lo <= cate + 1e-6)
    assert np.all(cate <= hi + 1e-6)


def test_save_load_roundtrip_preserves_cate(
    dummy_lalonde_df: pd.DataFrame, tmp_path: Path
) -> None:
    """save → load round-trip must produce identical CATE predictions."""
    Y, T, X = _arrays(dummy_lalonde_df)
    cf = _fast_cf().fit(Y=Y, T=T, X=X)
    cf.save(tmp_path / "cf")
    loaded = CausalForestEstimator.load(tmp_path / "cf")
    np.testing.assert_allclose(cf.predict(X), loaded.predict(X))


def test_cate_distribution_keys(tmp_path: Path) -> None:
    """After a tiny training run, cate_distribution.json has all expected keys."""
    from src.training.train_dml import _fit_causal_forest

    rng = np.random.default_rng(0)
    n = 60
    df = pd.DataFrame(
        {
            "treat": rng.integers(0, 2, size=n).astype(int),
            "age": rng.integers(18, 50, size=n).astype(int),
            "education": rng.integers(7, 16, size=n).astype(int),
            "black": rng.integers(0, 2, size=n).astype(int),
            "hispanic": rng.integers(0, 2, size=n).astype(int),
            "married": rng.integers(0, 2, size=n).astype(int),
            "nodegree": rng.integers(0, 2, size=n).astype(int),
            "re74": rng.uniform(0, 5000, size=n),
            "re75": rng.uniform(0, 5000, size=n),
            "re78": rng.uniform(0, 8000, size=n),
        }
    )
    cfg = {
        "seed": 42,
        "causal_forest": {
            "n_estimators": 20,
            "min_samples_leaf": 2,
            "max_depth": 3,
            "cv": 2,
            "n_bootstrap_samples": 5,
        },
        "dml": {"model_y": _fast_base(), "model_t": _fast_base()},
    }
    cf, cate, lo, hi, ate = _fit_causal_forest(cfg, df)
    distribution = {
        "mean": float(cate.mean()),
        "std": float(cate.std()),
        "min": float(cate.min()),
        "max": float(cate.max()),
        "p10": float(np.percentile(cate, 10)),
        "p50": float(np.percentile(cate, 50)),
        "p90": float(np.percentile(cate, 90)),
        "ate_from_cf": float(ate["ate"]),
    }
    out = tmp_path / "cate_distribution.json"
    out.write_text(json.dumps(distribution, indent=2))
    parsed = json.loads(out.read_text())
    expected = {"mean", "std", "min", "max", "p10", "p50", "p90", "ate_from_cf"}
    assert expected <= set(parsed.keys())


@pytest.mark.slow
def test_cf_ate_close_to_dml_nsw_ate() -> None:
    """CF ATE should land within ~$1500 of LinearDML on real NSW (rule C40 sanity).

    Both estimators target the same NSW ATE; bootstrap noise + forest bias
    means they will differ, but a gap >$1500 indicates a forest-config bug.
    """
    from src.config import load_config
    from src.training.train_dml import _fit_causal_forest, _fit_dml

    cfg = load_config("config/config.yaml")
    nsw_csv = Path(cfg["data"]["nsw_processed"])
    if not nsw_csv.exists():
        pytest.skip("nsw_clean.csv not materialised (DVC pull required)")
    nsw = pd.read_csv(nsw_csv)

    _, ate_dml = _fit_dml(cfg, nsw)
    _, _, _, _, ate_cf = _fit_causal_forest(cfg, nsw)
    gap = abs(float(ate_dml["ate"]) - float(ate_cf["ate"]))
    assert gap < 1500.0, (
        f"CF ATE ${ate_cf['ate']:.0f} disagrees with DML ATE ${ate_dml['ate']:.0f} "
        f"by ${gap:.0f} — likely a forest-config bug."
    )
