"""Tests for src/training/train_dml.py — smoke + real-NSW recovery (rule C37)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.training.train_dml import main


def _write_config(
    tmp_path: Path,
    nsw_csv: Path,
    obs_csv: Path,
    rcl_truth: dict,
    tolerance: float = 5000.0,
) -> Path:
    """Build a self-contained config + results.json for an isolated train run."""
    models_dir = tmp_path / "models"
    figures_dir = tmp_path / "reports" / "figures"
    results_json = tmp_path / "reports" / "results.json"
    results_json.parent.mkdir(parents=True, exist_ok=True)
    results_json.write_text(json.dumps({"rcl_ground_truth": rcl_truth}, indent=2))

    cfg = {
        "seed": 42,
        "data": {
            "nsw_url": "n/a",
            "cps_url": "n/a",
            "nsw_raw": str(nsw_csv),
            "cps_raw": str(obs_csv),
            "nsw_processed": str(nsw_csv),
            "cps_processed": str(obs_csv),
            "checksum_suffix": ".sha256",
        },
        "ground_truth": {
            "rcl_ate": rcl_truth["ate"],
            "rcl_ci_low": rcl_truth["ci_lower"],
            "rcl_ci_high": rcl_truth["ci_upper"],
            "tolerance": tolerance,
        },
        "dml": {
            "cv": 2,
            "model_y": {
                "n_estimators": 20,
                "max_depth": 3,
                "min_child_samples": 1,
                "learning_rate": 0.1,
                "verbose": -1,
            },
            "model_t": {
                "n_estimators": 20,
                "max_depth": 3,
                "min_child_samples": 1,
                "learning_rate": 0.1,
                "verbose": -1,
            },
        },
        "causal_forest": {
            "n_estimators": 50,
            "min_samples_leaf": 5,
            "max_depth": 5,
            "cv": 2,
            "inference": "bootstrap",
            "n_bootstrap_samples": 50,
        },
        "monitoring": {"cate_drift_window_size": 100, "cate_psi_threshold": 0.2},
        "api": {
            "rate_limit_cate": "30/minute",
            "max_payload_mb": 1,
            "cors_origins": ["http://localhost"],
            "trusted_hosts": ["localhost"],
        },
        "policy": {"top_fraction": 0.3},
        "ui": {
            "glassmorphism": False,
            "stream_pipeline": False,
            "primary_color": "#000",
            "secondary_color": "#fff",
        },
        "mlflow": {
            "experiment_name": "test_p3",
            "tracking_uri": f"file:{tmp_path / 'mlruns'}",
        },
        "paths": {
            "models_dir": str(models_dir),
            "reports_dir": str(tmp_path / "reports"),
            "figures_dir": str(figures_dir),
            "results_json": str(results_json),
        },
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    return cfg_path


def _synthetic_lalonde(n: int = 80, seed: int = 0) -> pd.DataFrame:
    """Generate a tiny LaLonde-shaped synthetic frame with a true ATE ~ 1500."""
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 50, size=n)
    education = rng.integers(7, 16, size=n)
    black = rng.integers(0, 2, size=n)
    hispanic = rng.integers(0, 2, size=n)
    married = rng.integers(0, 2, size=n)
    nodegree = rng.integers(0, 2, size=n)
    re74 = rng.uniform(0, 5000, size=n)
    re75 = rng.uniform(0, 5000, size=n)
    treat = rng.integers(0, 2, size=n)
    re78 = 1500.0 * treat + 0.4 * re74 + 0.4 * re75 + rng.normal(0, 500, size=n)
    re78 = np.clip(re78, 0, None)
    return pd.DataFrame(
        {
            "treat": treat.astype(int),
            "age": age.astype(int),
            "education": education.astype(int),
            "black": black.astype(int),
            "hispanic": hispanic.astype(int),
            "married": married.astype(int),
            "nodegree": nodegree.astype(int),
            "re74": re74.astype(float),
            "re75": re75.astype(float),
            "re78": re78.astype(float),
        }
    )


def test_train_dml_smoke(tmp_path: Path) -> None:
    """End-to-end smoke test on synthetic LaLonde — no exceptions, files written."""
    nsw = _synthetic_lalonde(n=80, seed=1)
    obs = _synthetic_lalonde(n=120, seed=2)
    nsw_csv = tmp_path / "nsw.csv"
    obs_csv = tmp_path / "nsw_treated_plus_cps_controls.csv"
    nsw.to_csv(nsw_csv, index=False)
    obs.to_csv(obs_csv, index=False)
    rcl_truth = {"ate": 1500.0, "ci_lower": 500.0, "ci_upper": 2500.0, "se": 500.0}

    cfg = _write_config(tmp_path, nsw_csv, obs_csv, rcl_truth, tolerance=5000.0)
    main(str(cfg))

    assert (tmp_path / "models" / "dml_nsw" / "linear_dml.joblib").exists()
    assert (tmp_path / "models" / "dml_cps" / "linear_dml.joblib").exists()


def test_calibration_metadata_written(tmp_path: Path) -> None:
    """calibration_metadata.json declares estimator=LinearDML for Day 8 anti-leakage."""
    nsw = _synthetic_lalonde(n=80, seed=3)
    obs = _synthetic_lalonde(n=120, seed=4)
    nsw_csv = tmp_path / "nsw.csv"
    obs_csv = tmp_path / "nsw_treated_plus_cps_controls.csv"
    nsw.to_csv(nsw_csv, index=False)
    obs.to_csv(obs_csv, index=False)
    rcl_truth = {"ate": 1500.0, "ci_lower": 500.0, "ci_upper": 2500.0, "se": 500.0}

    cfg = _write_config(tmp_path, nsw_csv, obs_csv, rcl_truth, tolerance=5000.0)
    main(str(cfg))

    meta = json.loads((tmp_path / "models" / "calibration_metadata.json").read_text())
    assert meta["estimator"] == "LinearDML"
    assert meta["cv"] == 2
    assert meta["model_y"] == "LGBMRegressor"
    assert meta["model_t"] == "LGBMClassifier"


def test_ate_table_three_way_keys(tmp_path: Path) -> None:
    """results.json['ate_table'] exposes the headline three-way comparison keys."""
    nsw = _synthetic_lalonde(n=80, seed=5)
    obs = _synthetic_lalonde(n=120, seed=6)
    nsw_csv = tmp_path / "nsw.csv"
    obs_csv = tmp_path / "nsw_treated_plus_cps_controls.csv"
    nsw.to_csv(nsw_csv, index=False)
    obs.to_csv(obs_csv, index=False)
    rcl_truth = {"ate": 1500.0, "ci_lower": 500.0, "ci_upper": 2500.0, "se": 500.0}

    cfg = _write_config(tmp_path, nsw_csv, obs_csv, rcl_truth, tolerance=5000.0)
    main(str(cfg))

    results = json.loads((tmp_path / "reports" / "results.json").read_text())
    table = results["ate_table"]
    for key in ("dml_nsw", "dml_cps", "rcl_ground_truth"):
        assert key in table, f"missing ate_table key: {key}"


def test_ate_table_figure_written(tmp_path: Path) -> None:
    """ate_three_way_table.png lands under reports/figures/ after a train run."""
    nsw = _synthetic_lalonde(n=80, seed=7)
    obs = _synthetic_lalonde(n=120, seed=8)
    nsw_csv = tmp_path / "nsw.csv"
    obs_csv = tmp_path / "nsw_treated_plus_cps_controls.csv"
    nsw.to_csv(nsw_csv, index=False)
    obs.to_csv(obs_csv, index=False)
    rcl_truth = {"ate": 1500.0, "ci_lower": 500.0, "ci_upper": 2500.0, "se": 500.0}

    cfg = _write_config(tmp_path, nsw_csv, obs_csv, rcl_truth, tolerance=5000.0)
    main(str(cfg))

    figure = tmp_path / "reports" / "figures" / "ate_three_way_table.png"
    assert figure.exists()


@pytest.mark.slow
def test_dml_nsw_recovers_rcl_ground_truth() -> None:
    """Rule C37 headline test on real DVC-tracked NSW data.

    Real NSW (445 rows) + real config + the project's RCT ground truth in
    reports/results.json must yield a DML-NSW 95% CI that contains $1,794
    within the config tolerance. This is the central scientific claim of the
    project — a failure means rule C40 is being violated.
    """
    from src.config import load_config

    cfg = load_config("config/config.yaml")
    nsw_csv = Path(cfg["data"]["nsw_processed"])
    if not nsw_csv.exists():
        pytest.skip("nsw_clean.csv not materialised (DVC pull required)")
    main("config/config.yaml")

    results = json.loads(Path(cfg["paths"]["results_json"]).read_text())
    dml_nsw = results["ate_table"]["dml_nsw"]
    truth = float(results["rcl_ground_truth"]["ate"])
    tol = float(cfg["ground_truth"]["tolerance"])
    assert dml_nsw["ci_lower"] - tol <= truth <= dml_nsw["ci_upper"] + tol
