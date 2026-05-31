"""Coverage-regression invariants (strategy rule B).

These tests encode the causal-ML guarantees of the project as code. If the
estimator ever drifts away from its scientific narrative, CI goes red here
*before* a bad model ships. Do NOT loosen ``TOLERANCE`` or ``OLS_BIAS_MIN_GAP``
to make a failing test pass — a failure means the model, not the test, is wrong.
"""

import json
from pathlib import Path

import pytest

RESULTS = Path("reports/results.json")
CATE_DIST = Path("models/cate_distribution.json")
TOLERANCE = 1200.0  # rule C37 — NSW DML 95% CI must contain RCT truth ± this
OLS_BIAS_MIN_GAP = (
    500.0  # rule C40 — OLS-CPS-unadjusted must differ from RCT by >= this
)


@pytest.fixture(scope="module")
def results() -> dict:
    """Headline ATE/heterogeneity/policy results from the Day 3-5 pipeline."""
    if not RESULTS.exists():
        pytest.skip("reports/results.json not yet generated (Day 3 train_dml)")
    return json.loads(RESULTS.read_text())


@pytest.fixture(scope="module")
def cate_distribution() -> dict:
    """CATE distribution summary (mean/std/percentiles) from the causal forest."""
    if not CATE_DIST.exists():
        pytest.skip("models/cate_distribution.json not yet generated (Day 4)")
    return json.loads(CATE_DIST.read_text())


def test_dml_nsw_recovers_rcl_ground_truth(results: dict) -> None:
    """Rule C37: the headline gate. DML-NSW CI must contain the RCT truth."""
    t = results["ate_table"]
    truth = t["rcl_ground_truth"]["ate"]
    dml = t["dml_nsw"]
    gap_to_lo = truth - dml["ci_lower"]
    gap_to_hi = dml["ci_upper"] - truth
    assert gap_to_lo >= -TOLERANCE and gap_to_hi >= -TOLERANCE, (
        f"DML-NSW CI [{dml['ci_lower']:.0f}, {dml['ci_upper']:.0f}] does not "
        f"contain RCT truth ${truth:.0f} within ±${TOLERANCE:.0f} (rule C37)."
    )


def test_ols_cps_is_biased_away_from_truth(results: dict) -> None:
    """Rule C40: the bias demonstration. Naive OLS on CPS MUST differ from RCT.

    If they ever agree, the bias narrative collapses — investigate immediately
    (likely an off-by-one in the CPS construction, or accidental shuffling).
    """
    t = results["ate_table"]
    truth = t["rcl_ground_truth"]["ate"]
    ols = t["ols_cps_unadjusted"]["ate"]
    assert abs(truth - ols) > OLS_BIAS_MIN_GAP, (
        f"OLS-CPS-unadjusted ATE ${ols:.0f} is suspiciously close to RCT "
        f"truth ${truth:.0f}. If genuine, rule C40's narrative breaks."
    )


def test_dml_cps_recovers_truth_better_than_ols(results: dict) -> None:
    """DML on the same observational construction should beat naive OLS.

    That closer-to-truth gap is the whole point of double machine learning.
    """
    t = results["ate_table"]
    truth = t["rcl_ground_truth"]["ate"]
    ols_err = abs(t["ols_cps_unadjusted"]["ate"] - truth)
    dml_err = abs(t["dml_cps"]["ate"] - truth)
    assert dml_err < ols_err, (
        f"DML-CPS error ${dml_err:.0f} >= OLS-CPS error ${ols_err:.0f}. "
        "DML should be closer to truth than OLS on the same data."
    )


def test_cate_distribution_has_variation(cate_distribution: dict) -> None:
    """CATE shouldn't be a constant — that would mean no heterogeneity."""
    std = cate_distribution["std"]
    assert std > 50.0, (
        f"CATE std ${std:.0f} is too small — predicted effects are nearly "
        "constant, which contradicts the H in HTE."
    )


def test_policy_top30_beats_random_baseline(
    results: dict, cate_distribution: dict
) -> None:
    """Targeting the top 30% by CATE must lift effect above the population mean.

    If the policy mean exceeds the population mean, targeting carries signal —
    Day 5's policy curve reflects this.
    """
    p = results["policy_targeting"]
    pop_mean = cate_distribution["mean"]
    assert p["mean_cate_targeted"] > pop_mean, (
        f"Targeted mean CATE ${p['mean_cate_targeted']:.0f} does not exceed "
        f"population mean ${pop_mean:.0f} — policy targeting has no signal."
    )
