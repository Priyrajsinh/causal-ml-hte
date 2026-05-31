"""Anti-leakage provenance checks for the DML calibration metadata.

Rule C33 forbids a holdout train/test split for the causal model: ``LinearDML``
fits on the FULL NSW dataset and relies on internal cross-fitting (``cv>=2``,
rule C35) as its validation strategy. These tests assert the recorded
provenance still reflects that contract — a cheap tripwire against someone
silently re-introducing a split or swapping nuisance learners.
"""

import json
from pathlib import Path

META = Path("models/calibration_metadata.json")


def test_calibration_used_full_nsw_no_holdout() -> None:
    """Rule C33/C35: full-NSW fit, cross-fitting, correct nuisance learners."""
    meta = json.loads(META.read_text())
    assert meta["estimator"] == "LinearDML"
    assert meta["fitted_on"] == "nsw_clean.csv"
    assert meta["cv"] >= 2, "cv must be >= 2 for cross-fitting (rule C35)"
    assert (
        meta["n_samples_nsw"] >= 400
    ), "NSW sample suspiciously small — should be 445; check the loader."
    assert meta["model_y"] == "LGBMRegressor"
    assert meta["model_t"] == "LGBMClassifier"
