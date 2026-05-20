"""Project-wide exception hierarchy for causal-ml-hte."""


class ProjectBaseError(Exception):
    """Base error for the causal-ml-hte project."""


class ConfigError(ProjectBaseError):
    """Raised on bad YAML or missing required config keys."""


class DataLoadError(ProjectBaseError):
    """Raised when a data file cannot be loaded or parsed."""


class ChecksumError(DataLoadError):
    """Raised when a file's SHA-256 does not match the stored sidecar."""


class ModelNotFoundError(ProjectBaseError):
    """Raised when a saved model artefact cannot be found on disk."""


class PredictionError(ProjectBaseError):
    """Raised when a model prediction fails validation (NaN/inf/shape)."""


class CausalAssumptionError(ProjectBaseError):
    """Raised when a causal assumption (overlap, exchangeability, SUTVA) is violated."""
