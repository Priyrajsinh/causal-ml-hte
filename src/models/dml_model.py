"""LinearDML + LightGBM nuisance models — the core causal estimator (rule C38).

Implements Chernozhukov et al. (2018) Double/Debiased Machine Learning via the
EconML ``LinearDML`` wrapper. The K-fold cross-fitting (rule C35, ``cv >= 2``)
IS the validation strategy: there is no separate test split (rule C33).
"""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from econml.dml import LinearDML
from lightgbm import LGBMClassifier, LGBMRegressor

from src.exceptions import ModelNotFoundError, PredictionError
from src.logger import get_logger
from src.models.base import BaseMLModel

logger = get_logger(__name__)

COVARIATES = [
    "age",
    "education",
    "black",
    "hispanic",
    "married",
    "nodegree",
    "re74",
    "re75",
]


class CausalDMLEstimator(BaseMLModel):
    """LinearDML with LightGBM nuisance models. Rule C35: ``cv >= 2`` always.

    Fitting strategy (rule C33): NO test split. DML does internal K-fold
    cross-fitting; holding out a separate test set on NSW (445 rows) would
    leave too few samples for stable nuisance estimation.
    """

    def __init__(
        self,
        cv: int = 5,
        model_y_kwargs: dict[str, Any] | None = None,
        model_t_kwargs: dict[str, Any] | None = None,
        seed: int = 42,
    ) -> None:
        """Configure the DML estimator (does not fit)."""
        if cv < 2:
            raise ValueError("cv must be >= 2 for cross-fitting (rule C35).")
        self.cv = cv
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
        self.estimator: LinearDML | None = None
        self._fitted = False

    def _build(self) -> LinearDML:
        """Construct a fresh LinearDML with LightGBM nuisance models."""
        return LinearDML(
            model_y=LGBMRegressor(**self.model_y_kwargs),
            model_t=LGBMClassifier(**self.model_t_kwargs),
            discrete_treatment=True,
            cv=self.cv,
            random_state=self.seed,
        )

    def fit(self, Y: np.ndarray, T: np.ndarray, X: np.ndarray) -> "CausalDMLEstimator":
        """Fit DML on (Y, T, X) using cross-fitted LightGBM nuisance models."""
        self.estimator = self._build()
        self.estimator.fit(Y=Y, T=T, X=X, W=None)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return the CATE vector for each row of X (rule C38)."""
        if not self._fitted or self.estimator is None:
            raise PredictionError("CausalDMLEstimator not fitted.")
        return np.asarray(self.estimator.effect(X)).reshape(-1)

    def ate(self, X: np.ndarray, alpha: float = 0.05) -> dict[str, float]:
        """Average treatment effect over X with a (1-alpha) confidence interval."""
        if not self._fitted or self.estimator is None:
            raise PredictionError("CausalDMLEstimator not fitted.")
        ate_val = float(self.estimator.ate(X))
        ci = self.estimator.ate_interval(X, alpha=alpha)
        return {
            "ate": ate_val,
            "ci_lower": float(ci[0]),
            "ci_upper": float(ci[1]),
            "alpha": float(alpha),
        }

    def cate_intervals(
        self, X: np.ndarray, alpha: float = 0.05
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return per-row (lower, upper) CATE CI bands at level (1-alpha)."""
        if not self._fitted or self.estimator is None:
            raise PredictionError("CausalDMLEstimator not fitted.")
        lo, hi = self.estimator.effect_interval(X, alpha=alpha)
        return np.asarray(lo).reshape(-1), np.asarray(hi).reshape(-1)

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
            "safe_predict: n=%d cols=%d range=[%.2f,%.2f]",
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
        joblib.dump(self.estimator, dir_ / "linear_dml.joblib")
        joblib.dump(
            {
                "cv": self.cv,
                "seed": self.seed,
                "model_y_kwargs": self.model_y_kwargs,
                "model_t_kwargs": self.model_t_kwargs,
                "fitted": self._fitted,
            },
            dir_ / "linear_dml_meta.joblib",
        )

    @classmethod
    def load(cls, dir_: Path) -> "CausalDMLEstimator":
        """Re-hydrate a previously saved CausalDMLEstimator from dir_."""
        dir_ = Path(dir_)
        if not (dir_ / "linear_dml.joblib").exists():
            raise ModelNotFoundError(f"No linear_dml.joblib in {dir_}")
        meta = joblib.load(dir_ / "linear_dml_meta.joblib")  # nosec B301
        obj = cls(
            cv=meta["cv"],
            model_y_kwargs=meta["model_y_kwargs"],
            model_t_kwargs=meta["model_t_kwargs"],
            seed=meta["seed"],
        )
        obj.estimator = joblib.load(dir_ / "linear_dml.joblib")  # nosec B301
        obj._fitted = bool(meta.get("fitted", True))
        return obj
