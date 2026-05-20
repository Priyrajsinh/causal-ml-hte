"""Config loader: reads config/config.yaml and validates required keys."""

from pathlib import Path
from typing import Any

import yaml

from src.exceptions import ConfigError

_REQUIRED_TOP_KEYS = {
    "seed",
    "data",
    "ground_truth",
    "dml",
    "causal_forest",
    "monitoring",
    "api",
    "policy",
    "ui",
    "mlflow",
    "paths",
}


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and validate config/config.yaml. Raise ConfigError on any issue."""
    try:
        with open(str(path)) as fh:
            cfg = yaml.safe_load(fh)
    except Exception as exc:
        raise ConfigError(f"Cannot parse config at {path}: {exc}") from exc
    if not isinstance(cfg, dict):
        raise ConfigError(f"Config at {path} must be a YAML mapping, got {type(cfg)}")
    missing = _REQUIRED_TOP_KEYS - cfg.keys()
    if missing:
        raise ConfigError(f"Config missing required top-level keys: {sorted(missing)}")
    return cfg
