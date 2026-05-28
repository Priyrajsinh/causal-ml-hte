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
- Model: 200-tree `CausalForestDML` with `BootstrapInference` (500 samples).
- Ground truth (NSW RCT ATE): ~ **\$1,794** (95% CI: [\$550, \$3,113]).

Adjust the eight pre-treatment covariates and click **Predict CATE** to see
the predicted treatment effect, 95% bootstrap CI, and a plain-English
recommendation (TREAT / DEFER).
