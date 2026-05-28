"""Rolling-window CATE drift monitor + Prometheus instruments (rule C42).

The FastAPI ``/cate`` route calls ``CATEDriftMonitor.record(cate)`` on every
prediction. The monitor maintains a fixed-size rolling window of the most
recent predicted CATE values; once the window has at least 50 entries it
re-exports:

- ``cate_mean`` Gauge — rolling-window mean CATE.
- ``cate_std`` Gauge — rolling-window std CATE.
- ``cate_distribution_drift_total`` Counter — incremented whenever PSI
  between the rolling window and the baseline NSW CATE distribution exceeds
  the configured threshold.

Population Stability Index (PSI) is the standard MLOps drift score:

    PSI = sum_b (p_cur[b] - p_ref[b]) * ln(p_cur[b] / p_ref[b])

>0.1 = small drift, >0.2 = significant drift (alarm), >0.25 = major drift.
The threshold is configured via ``config.yaml -> monitoring.cate_psi_threshold``.
"""

from collections import deque
from threading import Lock

import numpy as np
from prometheus_client import Counter, Gauge

CATE_DRIFT_TOTAL = Counter(
    "cate_distribution_drift_total",
    "Times rolling CATE distribution diverged from baseline (PSI > threshold).",
)
CATE_MEAN_GAUGE = Gauge("cate_mean", "Rolling mean predicted CATE (USD).")
CATE_STD_GAUGE = Gauge("cate_std", "Rolling std predicted CATE (USD).")


def _psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    """Population Stability Index between two 1-D distributions.

    Bins are reference-quantile based (so each reference bin has equal mass).
    ``p_ref`` and ``p_cur`` are floored at ``1e-6`` to keep the log finite
    when a bin happens to receive zero current samples.
    """
    if len(reference) == 0 or len(current) == 0:
        return 0.0
    edges = np.quantile(reference, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    ref_h, _ = np.histogram(reference, bins=edges)
    cur_h, _ = np.histogram(current, bins=edges)
    ref_p = np.clip(ref_h / len(reference), 1e-6, None)
    cur_p = np.clip(cur_h / len(current), 1e-6, None)
    return float(((cur_p - ref_p) * np.log(cur_p / ref_p)).sum())


class CATEDriftMonitor:
    """Thread-safe rolling-window CATE drift monitor."""

    def __init__(
        self,
        window_size: int,
        psi_threshold: float,
        baseline_cates: np.ndarray,
    ) -> None:
        """Initialize with a fixed window and a baseline reference distribution."""
        self.window: deque[float] = deque(maxlen=window_size)
        self.psi_threshold = psi_threshold
        self.baseline = np.asarray(baseline_cates, dtype=float)
        self.lock = Lock()

    def record(self, cate: float) -> None:
        """Append one CATE prediction; refresh gauges + drift counter as needed."""
        with self.lock:
            self.window.append(float(cate))
            if len(self.window) >= 50:
                arr = np.array(self.window)
                CATE_MEAN_GAUGE.set(float(arr.mean()))
                CATE_STD_GAUGE.set(float(arr.std()))
                psi = _psi(self.baseline, arr)
                if psi > self.psi_threshold:
                    CATE_DRIFT_TOTAL.inc()
