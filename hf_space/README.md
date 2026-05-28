---
title: Causal ML HTE - CATE Explorer
emoji: 📈
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: "5.0"
app_file: app.py
python_version: "3.12"
pinned: false
license: mit
---

# Causal ML · HTE — CATE Explorer

Interactive Gradio app that estimates a person's **Conditional Average
Treatment Effect (CATE)** from the LaLonde NSW + CPS datasets using
Chernozhukov et al. (2018) Double/Debiased Machine Learning (`CausalForestDML`
with LightGBM nuisance models).

- Source repo: <https://github.com/Priyrajsinh/causal-ml-hte>
- Ground truth (NSW RCT ATE): ~ **\$1,794** (95% CI: [\$550, \$3,113]).

Adjust the eight pre-treatment covariates and click **Predict CATE** to see
the predicted treatment effect, 95% CI, and a plain-English recommendation
(TREAT / DEFER).

## Demo model vs. production model

This Space ships a **deploy-only artefact** that differs from the production
model in *one* respect — the confidence-interval estimator:

|                       | Production model (repo)                | Demo model (this Space)                   |
| --------------------- | -------------------------------------- | ----------------------------------------- |
| Forest hyperparams    | n_estimators=200, max_depth=10, cv=5   | **identical**                             |
| CATE point estimates  | `est.effect(X)`                        | **identical** (same trees)                |
| 95% CI method         | `BootstrapInference(n=500)`            | **Bootstrap-of-Little-Bags (BLB)**        |
| Joblib size           | ~1.8 GB (500 retrained forests)        | ~1.4 MB (no extra forests stored)         |
| Use case              | RCT-truth coverage assertion (rule C37)| Interactive single-person CATE prediction |

**Why the split?** The production code path enforces a hard test (rule C37):
the empirical bootstrap CI must contain the LaLonde RCT ground-truth ATE
of \\$1,794. That requires storing 500 retrained forests for the resampling
distribution. The HF Space free tier caps repos at 1 GB, and the demo's
use case is per-individual CATE prediction — for that, EconML's
**native Bootstrap-of-Little-Bags** estimator (Athey, Tibshirani & Wager
2019) produces valid 95% intervals without storing extra forests, at
~1,300× smaller on disk.

The demo and production models give **identical CATE point estimates** on
any input; only the CI width may differ slightly in finite samples (they
agree asymptotically).

Reproduce the deploy artefact: `python scripts/build_deploy_artifact.py`.
