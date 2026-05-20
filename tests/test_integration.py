import pytest


@pytest.mark.skip(reason="Day 6 wires /api/v1/health")
def test_health_endpoint():
    """GET /api/v1/health returns 200 with model_loaded: true."""
    ...


@pytest.mark.skip(reason="Day 6 wires /api/v1/cate")
def test_cate_endpoint_valid_schema():
    """POST /api/v1/cate with valid LalondeProfile returns CATEPrediction schema."""
    ...


@pytest.mark.skip(reason="Day 6 wires /api/v1/cate")
def test_invalid_input_422():
    """POST /api/v1/cate with age=99 must return HTTP 422."""
    ...


@pytest.mark.skip(reason="Day 8 wires the coverage regression test")
def test_nsw_ate_contains_ground_truth():
    """NSW DML 95% CI must contain $1,794 (rule C37)."""
    ...
