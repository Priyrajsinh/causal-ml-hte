import pandas as pd
import pytest


@pytest.fixture
def dummy_lalonde_df() -> pd.DataFrame:
    """Tiny, well-formed LaLonde DataFrame for unit tests."""
    return pd.DataFrame(
        {
            "treat": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            "age": [25, 30, 22, 35, 28, 40, 19, 33, 27, 45],
            "education": [10, 12, 9, 13, 11, 14, 8, 12, 10, 13],
            "black": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            "hispanic": [0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
            "married": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            "nodegree": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            "re74": [
                0.0,
                1500.0,
                0.0,
                3000.0,
                500.0,
                2000.0,
                0.0,
                4500.0,
                800.0,
                5000.0,
            ],
            "re75": [
                0.0,
                1800.0,
                0.0,
                3200.0,
                400.0,
                2200.0,
                0.0,
                4800.0,
                900.0,
                5500.0,
            ],
            "re78": [
                0.0,
                4500.0,
                800.0,
                6500.0,
                1200.0,
                5200.0,
                500.0,
                7500.0,
                2000.0,
                8500.0,
            ],
        }
    )


@pytest.fixture
def mock_config() -> dict:
    """Minimal config dict used by unit tests that need config structure."""
    return {
        "seed": 42,
        "dml": {
            "cv": 5,
            "model_y": {"n_estimators": 50, "verbose": -1},
            "model_t": {"n_estimators": 50, "verbose": -1},
        },
        "ground_truth": {"rcl_ate": 1794.0, "tolerance": 600.0},
        "policy": {"top_fraction": 0.30},
    }
