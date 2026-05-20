"""Abstract base class shared by all causal estimators in the project."""

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class BaseMLModel(ABC):
    """Abstract base for every model in the project (rule C38).

    For causal models, ``predict(X)`` returns the CATE vector and
    ``predict_proba`` raises NotImplementedError (probabilities are not
    meaningful for treatment-effect estimators).
    """

    @abstractmethod
    def fit(self, Y: np.ndarray, T: np.ndarray, X: np.ndarray) -> "BaseMLModel":
        """Fit the model on outcome Y, treatment T, covariates X."""
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return CATE for each row of X."""
        ...

    @abstractmethod
    def save(self, dir_: Path) -> None:
        """Persist the fitted model to dir_."""
        ...

    @classmethod
    @abstractmethod
    def load(cls, dir_: Path) -> "BaseMLModel":
        """Load a previously saved model from dir_."""
        ...
