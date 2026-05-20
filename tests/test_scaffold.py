import pytest


def test_logger_returns_logger():
    """get_logger must return a Logger with the requested name."""
    from src.logger import get_logger

    assert get_logger("x").name == "x"


def test_logger_idempotent():
    """Calling get_logger twice with the same name must not add duplicate handlers."""
    from src.logger import get_logger

    log = get_logger("dup_test")
    count_before = len(log.handlers)
    get_logger("dup_test")
    assert len(log.handlers) == count_before


def test_exceptions_hierarchy():
    """Exception subclass relationships must match rule C39 hierarchy."""
    from src.exceptions import (
        CausalAssumptionError,
        ChecksumError,
        DataLoadError,
        PredictionError,
        ProjectBaseError,
    )

    assert issubclass(DataLoadError, ProjectBaseError)
    assert issubclass(ChecksumError, DataLoadError)
    assert issubclass(PredictionError, ProjectBaseError)
    assert issubclass(CausalAssumptionError, ProjectBaseError)


def test_base_model_is_abstract():
    """BaseMLModel must be abstract and not directly instantiable."""
    from src.models.base import BaseMLModel

    with pytest.raises(TypeError):
        BaseMLModel()


def test_lalonde_profile_validation():
    """LalondeProfile must reject out-of-range age."""
    import pydantic

    from src.data.schemas import LalondeProfile

    LalondeProfile(
        age=25,
        education=12,
        black=1,
        hispanic=0,
        married=0,
        nodegree=1,
        re74=0.0,
        re75=0.0,
    )
    with pytest.raises(pydantic.ValidationError):
        LalondeProfile(
            age=99,
            education=12,
            black=1,
            hispanic=0,
            married=0,
            nodegree=1,
            re74=0.0,
            re75=0.0,
        )


def test_ate_report_defaults_match_ground_truth():
    """ATEReport default RCT ground-truth must be $1,794."""
    from src.data.schemas import ATEReport

    r = ATEReport(
        ols_cps_ate=-300.0,
        ols_cps_ci=(-800.0, 200.0),
        dml_cps_ate=1500.0,
        dml_cps_ci=(800.0, 2200.0),
        dml_nsw_ate=1800.0,
        dml_nsw_ci=(900.0, 2700.0),
    )
    assert r.rcl_ground_truth_ate == 1794.0


def test_config_load_valid(tmp_path):
    """load_config must parse a valid YAML and return a dict with required keys."""
    import yaml

    from src.config import load_config

    cfg_data = {
        "seed": 42,
        "data": {},
        "ground_truth": {},
        "dml": {},
        "causal_forest": {},
        "monitoring": {},
        "api": {},
        "policy": {},
        "ui": {},
        "mlflow": {},
        "paths": {},
    }
    p = tmp_path / "config.yaml"
    with open(str(p), "w") as fh:
        yaml.dump(cfg_data, fh)
    cfg = load_config(p)
    assert cfg["seed"] == 42


def test_config_load_missing_key(tmp_path):
    """load_config must raise ConfigError when a required key is absent."""
    import yaml

    from src.config import load_config
    from src.exceptions import ConfigError

    p = tmp_path / "bad.yaml"
    with open(str(p), "w") as fh:
        yaml.dump({"seed": 1}, fh)
    with pytest.raises(ConfigError):
        load_config(p)


def test_set_seed_runs_without_error():
    """set_seed must complete without raising, seeding Python + NumPy."""
    from src.utils.seed import set_seed

    set_seed(42)


def test_save_training_stats_and_check_skew(tmp_path):
    """save_training_stats writes JSON; check_skew flags out-of-range values."""
    import numpy as np
    import pandas as pd

    from src.data.skew_check import COVARIATES, check_skew, save_training_stats

    X = pd.DataFrame([[25, 10, 1, 0, 0, 1, 0.0, 0.0]] * 5, columns=COVARIATES)
    stats_path = tmp_path / "training_stats.json"
    save_training_stats(X, stats_path)
    assert stats_path.exists()

    in_range_row = np.array([25.0, 10.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    result = check_skew(in_range_row, stats_path)
    assert not any(result.values())

    out_of_range_row = np.array([99.0, 10.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    result_oob = check_skew(out_of_range_row, stats_path)
    assert result_oob["age"] is True
