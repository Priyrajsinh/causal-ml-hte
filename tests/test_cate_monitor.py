"""Unit tests for src/api/cate_monitor.py (rule C42)."""

from __future__ import annotations

import numpy as np

from src.api.cate_monitor import CATE_DRIFT_TOTAL, CATEDriftMonitor, _psi


def test_psi_zero_for_identical_distributions() -> None:
    """PSI of a distribution against itself is ~0."""
    rng = np.random.default_rng(0)
    arr = rng.normal(0, 1, size=1000)
    assert _psi(arr, arr) < 1e-6


def test_psi_positive_for_shifted_distributions() -> None:
    """PSI is strictly positive for a clearly-shifted current distribution."""
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, size=1000)
    cur = rng.normal(3, 1, size=1000)  # shifted +3 sigma
    assert _psi(ref, cur) > 0.5  # major drift territory


def test_psi_empty_returns_zero() -> None:
    """PSI degrades gracefully on empty input arrays."""
    assert _psi(np.array([]), np.array([1.0, 2.0])) == 0.0
    assert _psi(np.array([1.0, 2.0]), np.array([])) == 0.0


def test_monitor_records_into_rolling_window() -> None:
    """``record`` appends to the deque and respects ``maxlen``."""
    baseline = np.random.default_rng(0).normal(1800, 1200, size=500)
    monitor = CATEDriftMonitor(
        window_size=100, psi_threshold=0.2, baseline_cates=baseline
    )
    for v in range(120):
        monitor.record(float(v))
    assert len(monitor.window) == 100
    # Oldest 20 were evicted, so the deque starts at 20
    assert monitor.window[0] == 20.0


def test_monitor_increments_drift_counter_on_shifted_distribution() -> None:
    """Feeding clearly-shifted CATEs increments ``cate_distribution_drift_total``."""
    baseline = np.random.default_rng(0).normal(1800, 1200, size=500)
    monitor = CATEDriftMonitor(
        window_size=200, psi_threshold=0.2, baseline_cates=baseline
    )

    # Snapshot the counter before
    samples_before = list(CATE_DRIFT_TOTAL.collect())[0].samples
    before = next((s.value for s in samples_before if s.name.endswith("_total")), 0.0)

    shifted = np.random.default_rng(1).normal(20000, 500, size=200)
    for v in shifted:
        monitor.record(float(v))

    samples_after = list(CATE_DRIFT_TOTAL.collect())[0].samples
    after = next((s.value for s in samples_after if s.name.endswith("_total")), 0.0)

    assert after > before


def test_monitor_does_not_increment_before_50_records() -> None:
    """No drift counter increments until the rolling window has >= 50 entries."""
    baseline = np.random.default_rng(0).normal(1800, 1200, size=500)
    monitor = CATEDriftMonitor(
        window_size=200, psi_threshold=0.2, baseline_cates=baseline
    )

    samples_before = list(CATE_DRIFT_TOTAL.collect())[0].samples
    before = next((s.value for s in samples_before if s.name.endswith("_total")), 0.0)

    # Only 40 wildly-shifted records — should not arm the drift check
    for v in range(40):
        monitor.record(99999.0)

    samples_after = list(CATE_DRIFT_TOTAL.collect())[0].samples
    after = next((s.value for s in samples_after if s.name.endswith("_total")), 0.0)

    assert after == before
