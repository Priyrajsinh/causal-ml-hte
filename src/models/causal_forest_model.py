"""CausalForestDML wrapper — heterogeneous treatment-effect estimator (rule C38).

Athey & Wager (2018) "Estimation and Inference of Heterogeneous Treatment
Effects using Random Forests" — *honest* splitting (data used to choose a
split is not used to estimate the leaf treatment effect). EconML's
``CausalForestDML`` wires this into the DML cross-fitting framework so the
forest is doubly robust.

Inference is bootstrap-based: ``BootstrapInference`` resamples the training
set + retrains the forest, repeating ``n_bootstrap_samples`` times. Slow but
principled — see ``@pytest.mark.slow`` tests for the on-real-data check.
"""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from econml.dml import CausalForestDML
from econml.inference import BootstrapInference
from lightgbm import LGBMClassifier, LGBMRegressor

from src.exceptions import ModelNotFoundError, PredictionError
from src.logger import get_logger
from src.models.base import BaseMLModel
from src.models.dml_model import COVARIATES

logger = get_logger(__name__)


class CausalForestEstimator(BaseMLModel):
    """``CausalForestDML`` + LightGBM nuisance + bootstrap inference."""

    def __init__(
        self,
        n_estimators: int = 200,
        min_samples_leaf: int = 10,
        max_depth: int = 10,
        cv: int = 5,
        n_bootstrap_samples: int = 500,
        model_y_kwargs: dict[str, Any] | None = None,
        model_t_kwargs: dict[str, Any] | None = None,
        seed: int = 42,
    ) -> None:
        """Configure the forest (does not fit)."""
        if cv < 2:
            raise ValueError("cv must be >= 2 for cross-fitting (rule C35).")
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.cv = cv
        self.n_bootstrap_samples = n_bootstrap_samples
        self.seed = seed
        self.model_y_kwargs: dict[str, Any] = model_y_kwargs or {
            "n_estimators": 200,
            "max_depth": 4,
            "learning_rate": 0.05,
            "min_child_samples": 5,
            "verbose": -1,
            "random_state": seed,
        }
        self.model_t_kwargs: dict[str, Any] = model_t_kwargs or {
            "n_estimators": 200,
            "max_depth": 4,
            "learning_rate": 0.05,
            "min_child_samples": 5,
            "verbose": -1,
            "random_state": seed,
        }
        self.estimator: CausalForestDML | None = None
        self._fitted = False

    def _build(self) -> CausalForestDML:
        """Construct a fresh CausalForestDML with LightGBM nuisance models."""
        return CausalForestDML(
            n_estimators=self.n_estimators,
            min_samples_leaf=self.min_samples_leaf,
            max_depth=self.max_depth,
            discrete_treatment=True,
            cv=self.cv,
            model_y=LGBMRegressor(**self.model_y_kwargs),
            model_t=LGBMClassifier(**self.model_t_kwargs),
            random_state=self.seed,
        )

    def fit(
        self, Y: np.ndarray, T: np.ndarray, X: np.ndarray
    ) -> "CausalForestEstimator":
        """Fit the causal forest with normal-approximation bootstrap inference.

        ``bootstrap_type='normal'`` centres the CI on the point estimate by
        construction (``theta_hat ± z * sd(theta*)``), so the per-row guarantee
        ``ci_lower <= cate <= ci_upper`` holds for every prediction.
        """
        self.estimator = self._build()
        self.estimator.fit(
            Y=Y,
            T=T,
            X=X,
            inference=BootstrapInference(
                n_bootstrap_samples=self.n_bootstrap_samples,
                n_jobs=1,
                bootstrap_type="normal",
            ),
        )
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return per-row CATE (rule C38)."""
        if not self._fitted or self.estimator is None:
            raise PredictionError("CausalForestEstimator not fitted.")
        return np.asarray(self.estimator.effect(X)).reshape(-1)

    def predict_with_ci(
        self, X: np.ndarray, alpha: float = 0.05
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (cate, ci_lower, ci_upper) per row at level (1 - alpha).

        Routes through ``safe_predict`` so NaN/inf/shape guards (rule C36)
        gate every CATE inference call.
        """
        if not self._fitted or self.estimator is None:
            raise PredictionError("CausalForestEstimator not fitted.")
        cate = self.safe_predict(X)
        lo, hi = self.estimator.effect_interval(X, alpha=alpha)
        return cate, np.asarray(lo).reshape(-1), np.asarray(hi).reshape(-1)

    def ate(self, X: np.ndarray, alpha: float = 0.05) -> dict[str, float]:
        """Average treatment effect with a (1 - alpha) confidence interval."""
        if not self._fitted or self.estimator is None:
            raise PredictionError("CausalForestEstimator not fitted.")
        ate_val = float(self.estimator.ate(X))
        ci = self.estimator.ate_interval(X, alpha=alpha)
        return {
            "ate": ate_val,
            "ci_lower": float(ci[0]),
            "ci_upper": float(ci[1]),
            "alpha": float(alpha),
        }

    def safe_predict(self, X: np.ndarray) -> np.ndarray:
        """Rule C36 — NaN / inf / shape guards before delegating to predict."""
        if not self._fitted:
            raise PredictionError("Model not fitted.")
        X = np.asarray(X)
        if X.ndim != 2:
            raise PredictionError(f"Expected 2-D X, got shape {X.shape}.")
        if X.shape[1] != len(COVARIATES):
            raise PredictionError(
                f"Expected {len(COVARIATES)} covariates, got {X.shape[1]}."
            )
        if np.isnan(X).any() or np.isinf(X).any():
            raise PredictionError("Input contains NaN or inf.")
        logger.info(
            "safe_predict[cf]: n=%d cols=%d range=[%.2f,%.2f]",
            X.shape[0],
            X.shape[1],
            float(X.min()),
            float(X.max()),
        )
        return self.predict(X)

    def save(self, dir_: Path) -> None:
        """Persist the fitted estimator + meta to dir_ via joblib."""
        dir_ = Path(dir_)
        dir_.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.estimator, dir_ / "causal_forest.joblib")
        joblib.dump(
            {
                "n_estimators": self.n_estimators,
                "min_samples_leaf": self.min_samples_leaf,
                "max_depth": self.max_depth,
                "cv": self.cv,
                "n_bootstrap_samples": self.n_bootstrap_samples,
                "seed": self.seed,
                "model_y_kwargs": self.model_y_kwargs,
                "model_t_kwargs": self.model_t_kwargs,
                "fitted": self._fitted,
            },
            dir_ / "causal_forest_meta.joblib",
        )

    @classmethod
    def load(cls, dir_: Path) -> "CausalForestEstimator":
        """Re-hydrate a previously saved CausalForestEstimator from dir_."""
        dir_ = Path(dir_)
        if not (dir_ / "causal_forest.joblib").exists():
            raise ModelNotFoundError(f"No causal_forest.joblib in {dir_}")
        meta = joblib.load(dir_ / "causal_forest_meta.joblib")  # nosec B301
        obj = cls(
            n_estimators=meta["n_estimators"],
            min_samples_leaf=meta["min_samples_leaf"],
            max_depth=meta["max_depth"],
            cv=meta["cv"],
            n_bootstrap_samples=meta["n_bootstrap_samples"],
            seed=meta["seed"],
            model_y_kwargs=meta["model_y_kwargs"],
            model_t_kwargs=meta["model_t_kwargs"],
        )
        obj.estimator = joblib.load(dir_ / "causal_forest.joblib")  # nosec B301
        obj._fitted = bool(meta.get("fitted", True))
        return obj
