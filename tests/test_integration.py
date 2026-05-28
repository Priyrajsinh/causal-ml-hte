"""FastAPI integration tests (Day 6).

These hit the live ASGI app via Starlette's ``TestClient`` and require the
1.8 GB causal-forest joblib in ``models/causal_forest/``. Marked ``slow`` so
they are opt-in (default pytest skips them via ``addopts = "-m 'not slow'"``);
GH CI does not ship the joblib artefact, so the marker doubles as a CI guard.

Run locally:

    pytest tests/test_integration.py -m slow -v
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.slow


def _model_artefact() -> Path:
    """Return the path to the causal forest joblib (skip-trigger if missing)."""
    return Path("models/causal_forest/causal_forest.joblib")


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    """Module-scoped TestClient — lifespan fires once, joblib loaded once."""
    if not _model_artefact().exists():
        pytest.skip("causal_forest.joblib not materialised — run `make train` first")
    from src.api.app import app

    with TestClient(app) as c:
        yield c


def test_health_endpoint(client: TestClient) -> None:
    """GET /api/v1/health returns 200 with all required keys."""
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["model_loaded"] is True
    for key in {
        "status",
        "model_loaded",
        "uptime_seconds",
        "memory_mb",
        "version",
        "n_cate_predictions_served",
    }:
        assert key in body


def test_cate_endpoint_valid_schema(client: TestClient) -> None:
    """POST /api/v1/cate with a valid LalondeProfile returns CATEPrediction."""
    payload = {
        "age": 25,
        "education": 10,
        "black": 1,
        "hispanic": 0,
        "married": 0,
        "nodegree": 1,
        "re74": 0.0,
        "re75": 0.0,
    }
    r = client.post("/api/v1/cate", json=payload)
    assert r.status_code == 200
    body = r.json()
    for k in {"cate", "ci_lower", "ci_upper", "recommendation", "nl_summary"}:
        assert k in body
    assert body["ci_lower"] <= body["cate"] <= body["ci_upper"]
    assert body["recommendation"] in {"TREAT", "DEFER"}


def test_invalid_input_422(client: TestClient) -> None:
    """Out-of-range age must return HTTP 422 from Pydantic."""
    bad = {
        "age": 999,
        "education": 10,
        "black": 0,
        "hispanic": 0,
        "married": 0,
        "nodegree": 0,
        "re74": 0.0,
        "re75": 0.0,
    }
    r = client.post("/api/v1/cate", json=bad)
    assert r.status_code == 422


def test_ate_comparison_returns_full_table(client: TestClient) -> None:
    """GET /api/v1/ate_comparison returns the OLS/DML/RCT comparison."""
    r = client.get("/api/v1/ate_comparison")
    assert r.status_code == 200
    body = r.json()
    for k in {"ols_cps_ate", "dml_cps_ate", "dml_nsw_ate", "rcl_ground_truth_ate"}:
        assert k in body
    # Rule C40 sanity: DML on NSW is close to the RCT truth; OLS on CPS is biased.
    assert abs(body["dml_nsw_ate"] - body["rcl_ground_truth_ate"]) < 1500
    assert abs(body["ols_cps_ate"] - body["rcl_ground_truth_ate"]) > 500


def test_metrics_exposes_cate_drift_counter(client: TestClient) -> None:
    """GET /metrics exposes the CATE drift Counter + Gauges."""
    r = client.get("/metrics")
    assert r.status_code == 200
    text = r.text
    assert "cate_distribution_drift_total" in text
    assert "cate_mean" in text


def test_model_info_returns_results_json(client: TestClient) -> None:
    """GET /api/v1/model_info returns the full results.json (incl. SHAP)."""
    r = client.get("/api/v1/model_info")
    assert r.status_code == 200
    body = r.json()
    assert "ate_table" in body
    assert "shap_moderators" in body
