"""Build a small ``CausalForestDML`` artefact for the HF Space demo.

WHY THIS SCRIPT EXISTS — the demo-vs-production split
=====================================================

The **production** CausalForestEstimator (see ``src/models/causal_forest_model.py``)
fits with ``BootstrapInference(n_bootstrap_samples=500)`` because rule C37
requires the empirical bootstrap CI to contain the RCT ground-truth ATE of
\\$1,794. That makes the saved joblib ~1.8 GB — it stores the original 200-tree
forest **plus 500 retrained bootstrap forests** used for the resampling
distribution.

The **HF Space demo** does not need bootstrap CIs. Hosted free-tier Spaces
cap repos at 1 GB, and the demo's use case is *single-individual interactive
CATE prediction with a 95% interval bar*, not a one-shot population-level
ATE coverage assertion. EconML's ``CausalForestDML`` exposes its own native
inference method — **Bootstrap-of-Little-Bags (BLB)** (Athey, Tibshirani &
Wager 2019) — which is computed from the forest's honest splits *without*
storing extra retrained forests. ``effect_interval(X, alpha=0.05)`` works
identically.

So this artefact is:

- *Same forest hyperparameters* as production (n_estimators=200,
  max_depth=10, min_samples_leaf=10, cv=5) -> CATE point estimates and
  per-row rankings are unchanged.
- *Different inference method*: BLB (native, in-forest) instead of
  ``BootstrapInference`` (external resampling). The two are valid CI
  methods on causal forests; they will differ in finite-sample CI width
  but agree asymptotically.
- *Much smaller on disk*: ~10-100 MB instead of ~1.8 GB.

Run::

    venv/Scripts/python scripts/build_deploy_artifact.py

Output::

    hf_space/causal_forest.joblib   # <1 GB, BLB inference
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import joblib
import pandas as pd
from econml.dml import CausalForestDML
from lightgbm import LGBMClassifier, LGBMRegressor

# Allow direct invocation: `python scripts/build_deploy_artifact.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config  # noqa: E402
from src.data.validation import validate_lalonde_df  # noqa: E402
from src.logger import get_logger  # noqa: E402
from src.models.dml_model import COVARIATES  # noqa: E402

logger = get_logger(__name__)


def main(config_path: str = "config/config.yaml") -> None:
    """Fit a BLB-inference CausalForestDML and write it to hf_space/."""
    cfg = load_config(config_path)
    nsw = validate_lalonde_df(pd.read_csv(cfg["data"]["nsw_processed"]))

    Y = nsw["re78"].to_numpy(dtype=float)
    T = nsw["treat"].to_numpy(dtype=int)
    X = nsw[COVARIATES].to_numpy(dtype=float)

    seed = int(cfg["seed"])
    cf = cfg["causal_forest"]
    lgbm_y_kwargs = {**cfg["dml"]["model_y"], "random_state": seed}
    lgbm_t_kwargs = {**cfg["dml"]["model_t"], "random_state": seed}

    est = CausalForestDML(
        n_estimators=int(cf["n_estimators"]),
        min_samples_leaf=int(cf["min_samples_leaf"]),
        max_depth=int(cf["max_depth"]),
        discrete_treatment=True,
        cv=int(cf["cv"]),
        model_y=LGBMRegressor(**lgbm_y_kwargs),
        model_t=LGBMClassifier(**lgbm_t_kwargs),
        random_state=seed,
    )

    logger.info("Fitting deploy CausalForestDML (BLB inference, no BootstrapInference)")
    t0 = time.time()
    est.fit(Y=Y, T=T, X=X)  # default inference -> BLB
    logger.info("Fit complete in %.1fs", time.time() - t0)

    out = Path("hf_space/causal_forest.joblib")
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(est, str(out), compress=("gzip", 3))

    size_mb = out.stat().st_size / (1024 * 1024)
    logger.info("Wrote %s (%.1f MB, gzip-3 compressed)", out, size_mb)
    if size_mb > 950:
        logger.warning("Artefact is %.0f MB - close to HF free-tier 1 GB cap.", size_mb)

    # Smoke check: predict CATE + CI on the first row
    cate = float(est.effect(X[:1])[0])
    lo, hi = est.effect_interval(X[:1], alpha=0.05)
    logger.info(
        "Smoke check row 0: CATE=$%.0f, 95%% CI=[$%.0f, $%.0f]",
        cate,
        float(lo[0]),
        float(hi[0]),
    )


if __name__ == "__main__":
    main()
