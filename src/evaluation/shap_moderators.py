"""SHAP analysis on the CausalForestDML — moderators of treatment effect.

KEY DISTINCTION (the intellectual core of this module):

- SHAP on a predictive model (e.g. RandomForest predicting re78) → identifies
  *outcome drivers*: which features explain 1978 earnings overall?
- SHAP on a causal forest (predicting CATE) → identifies *moderators*: which
  features explain *who benefits more from treatment*?

These are fundamentally different questions. ``re74`` may be a strong driver
of ``re78`` but flat as a moderator — knowing someone's 1974 earnings tells us
their 1978 baseline, but might not tell us how much *additional* benefit they
get from job training. Education predicts both earnings AND treatment-effect
size in LaLonde NSW, so it is both a driver and a moderator.

We SHAP-explain the causal forest's CATE predictions, not the outcome.

``CausalForestDML`` is a forest-of-forests with internal cross-fitting; SHAP's
``TreeExplainer`` cannot consume that structure directly, so we use the
model-agnostic ``KernelExplainer`` with the forest's ``.effect(X)`` as the
black-box CATE callable.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.logger import get_logger
from src.models.dml_model import COVARIATES

plt.switch_backend("Agg")
logger = get_logger(__name__)


def build_cate_explainer(
    causal_forest_estimator: object,
    X_background: np.ndarray,
    background_size: int = 50,
    seed: int = 42,
) -> shap.KernelExplainer:
    """Wrap ``cf.effect(X)`` in a ``shap.KernelExplainer``.

    A small k-sample background set keeps KernelExplainer tractable on the
    NSW data scale (445 rows). ``effect`` returns one CATE per row.
    """

    def cate_fn(X: np.ndarray) -> np.ndarray:
        """Black-box CATE callable: forwards X to the causal forest."""
        effect = causal_forest_estimator.effect  # type: ignore[attr-defined]
        return np.asarray(effect(X)).reshape(-1)

    bg_df = pd.DataFrame(X_background, columns=COVARIATES)
    bg = shap.sample(bg_df, background_size, random_state=seed)
    return shap.KernelExplainer(cate_fn, bg)


def explain_cate(
    causal_forest_estimator: object,
    X: np.ndarray,
    n_samples: int = 100,
    n_explain: int | None = None,
    seed: int = 42,
) -> shap.Explanation:
    """Compute SHAP values explaining CATE predictions on X.

    KernelExplainer is O(rows × n_samples × features). For NSW the typical
    budget is 80 rows × 100 samples × 8 features — under one minute on CPU.
    """
    X = np.asarray(X, dtype=float)
    if n_explain is not None and len(X) > n_explain:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X), size=n_explain, replace=False)
        X_sub = X[idx]
    else:
        X_sub = X

    explainer = build_cate_explainer(causal_forest_estimator, X_background=X, seed=seed)
    shap_values = explainer.shap_values(
        pd.DataFrame(X_sub, columns=COVARIATES),
        nsamples=n_samples,
    )
    logger.info(
        "explain_cate: explained n=%d rows, n_samples=%d, |features|=%d",
        X_sub.shape[0],
        n_samples,
        X_sub.shape[1],
    )
    return shap.Explanation(
        values=np.asarray(shap_values),
        base_values=np.full(X_sub.shape[0], float(explainer.expected_value)),
        data=X_sub,
        feature_names=COVARIATES,
    )


def plot_shap_beeswarm(explanation: shap.Explanation, out_path: Path) -> None:
    """Global moderator importance — beeswarm.

    Each dot is one individual. X-axis is the SHAP value: how much *this
    feature* shifts the predicted CATE for *this individual*, relative to the
    average CATE. Color is the feature's raw value. Features with the widest
    horizontal spread are the strongest moderators — they explain the biggest
    share of *who* benefits from treatment.
    """
    fig = plt.figure(figsize=(9, 6))
    shap.plots.beeswarm(explanation, max_display=8, show=False)
    plt.title("Moderators of treatment effect (SHAP on CausalForestDML)")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_shap_bar(explanation: shap.Explanation, out_path: Path) -> None:
    """Mean ``|SHAP|`` per feature — moderator-ranking bar chart."""
    fig = plt.figure(figsize=(8, 5))
    shap.plots.bar(explanation, max_display=8, show=False)
    plt.title("Mean |SHAP| — moderator importance")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_shap_waterfall_individual(
    explanation: shap.Explanation, idx: int, out_path: Path
) -> None:
    """For one individual: why is their predicted CATE above or below average?"""
    fig = plt.figure(figsize=(9, 5))
    shap.plots.waterfall(explanation[idx], max_display=8, show=False)
    plt.title(f"Individual {idx}: SHAP decomposition of predicted CATE")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def save_moderator_summary(
    explanation: shap.Explanation, out_path: Path
) -> dict[str, float]:
    """Persist mean |SHAP| per feature, sorted descending, for ``results.json``."""
    mean_abs = np.abs(explanation.values).mean(axis=0)
    summary: dict[str, float] = {f: float(s) for f, s in zip(COVARIATES, mean_abs)}
    summary = dict(sorted(summary.items(), key=lambda kv: -kv[1]))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(out_path), "w") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("Moderator summary written -> %s", out_path)
    return summary
