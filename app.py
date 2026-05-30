"""Streamlit 4-tab dashboard — Causal ML · HTE (Day 7).

Tabs:
  1 · ATE Comparison  — OLS-bias vs DML-recovery vs RCT ground truth
  2 · CATE Explorer   — 8 sliders → CATE + CI + SHAP waterfall
  3 · Heterogeneity   — who benefits more? quartile bars + scatters
  4 · Policy Targeting — top-30% policy curve + lift estimate
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

from src.api.nl_translator import translate_cate
from src.config import load_config
from src.data.schemas import LalondeProfile
from src.evaluation.shap_moderators import build_cate_explainer
from src.models.causal_forest_model import CausalForestEstimator
from src.models.dml_model import COVARIATES

plt.switch_backend("Agg")

st.set_page_config(
    page_title="Causal ML · HTE",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📈",
)

_CSS_PATH = Path(__file__).parent / "src" / "api" / "streamlit_glass.css"
with open(str(_CSS_PATH)) as _fh:
    st.markdown(f"<style>{_fh.read()}</style>", unsafe_allow_html=True)


@st.cache_resource
def load_artifacts() -> tuple:
    """Load causal forest, CATE frame, results JSON, and SHAP explainer."""
    cfg = load_config("config/config.yaml")
    cf = CausalForestEstimator.load(Path(cfg["paths"]["models_dir"]) / "causal_forest")
    cate_df = pd.read_parquet(
        Path(cfg["paths"]["reports_dir"]) / "cate_per_row.parquet"
    )
    results = json.loads(Path(cfg["paths"]["results_json"]).read_text())
    explainer = build_cate_explainer(
        cf.estimator,
        X_background=cate_df[COVARIATES].to_numpy(),
    )
    return cfg, cf, cate_df, results, explainer


cfg, cf, cate_df, results, explainer = load_artifacts()

# ── Hero ──────────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="hero">
  <h1>Causal ML · Heterogeneous Treatment Effects</h1>
  <p>LaLonde NSW + CPS &nbsp;·&nbsp; Double ML (Chernozhukov 2018)
     &nbsp;·&nbsp; CausalForestDML &nbsp;·&nbsp; SHAP moderators</p>
  <div class="hero-links">
    <a href="https://huggingface.co/spaces/Priyrajsinh/causal-ml-hte-cate-explorer"
       target="_blank">Live Gradio Space</a>
    <a href="https://github.com/Priyrajsinh/causal-ml-hte"
       target="_blank">GitHub</a>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📊 ATE Comparison (OLS vs DML vs RCT)",
        "🔍 CATE Explorer + SHAP",
        "👥 Heterogeneity",
        "🎯 Policy Targeting",
    ]
)

with tab1:
    st.subheader("Naive OLS hides the truth; Double ML recovers it.")
    st.markdown(
        """
The **LaLonde NSW** dataset is a *randomised control trial* — random
assignment guarantees the difference of group means is an unbiased ATE.
That number is **~$1,794** in 1978 earnings.

The **LaLonde-CPS** construction replaces NSW controls with observational
CPS controls. Treated and control groups now differ systematically —
unemployed participants vs employed CPS workers. **Naive OLS on this
construction yields a strongly biased ATE** (often negative, i.e. job
training appears to *hurt* earnings). This bias *is* the story.

**Double ML (Chernozhukov 2018)** — Robinson decomposition + 5-fold
cross-fitting with LightGBM nuisance models — recovers an ATE close to
the RCT truth even on the observationally confounded data.
"""
    )

    t = results["ate_table"]
    df_ate = pd.DataFrame(
        [
            {
                "Estimator": "OLS · CPS (unadjusted)",
                "ATE ($)": t["ols_cps_unadjusted"]["ate"],
                "95% CI lower": t["ols_cps_unadjusted"]["ci_lower"],
                "95% CI upper": t["ols_cps_unadjusted"]["ci_upper"],
            },
            {
                "Estimator": "OLS · CPS (adjusted)",
                "ATE ($)": t["ols_cps_adjusted"]["ate"],
                "95% CI lower": t["ols_cps_adjusted"]["ci_lower"],
                "95% CI upper": t["ols_cps_adjusted"]["ci_upper"],
            },
            {
                "Estimator": "DML · CPS",
                "ATE ($)": t["dml_cps"]["ate"],
                "95% CI lower": t["dml_cps"]["ci_lower"],
                "95% CI upper": t["dml_cps"]["ci_upper"],
            },
            {
                "Estimator": "DML · NSW",
                "ATE ($)": t["dml_nsw"]["ate"],
                "95% CI lower": t["dml_nsw"]["ci_lower"],
                "95% CI upper": t["dml_nsw"]["ci_upper"],
            },
            {
                "Estimator": "RCT ground truth (NSW)",
                "ATE ($)": t["rcl_ground_truth"]["ate"],
                "95% CI lower": t["rcl_ground_truth"]["ci_lower"],
                "95% CI upper": t["rcl_ground_truth"]["ci_upper"],
            },
        ]
    )
    st.dataframe(
        df_ate.style.format(
            {
                "ATE ($)": "${:,.0f}",
                "95% CI lower": "${:,.0f}",
                "95% CI upper": "${:,.0f}",
            }
        ),
        use_container_width=True,
    )

    # Inline bar chart with RCT reference line
    _ates = [row["ATE ($)"] for row in df_ate.to_dict("records")]
    _ci_lo = [row["95% CI lower"] for row in df_ate.to_dict("records")]
    _ci_hi = [row["95% CI upper"] for row in df_ate.to_dict("records")]
    _labels = [
        "OLS·CPS\n(unadj.)",
        "OLS·CPS\n(adj.)",
        "DML·CPS",
        "DML·NSW",
        "RCT\ntruth",
    ]
    _colors = ["#ef4444", "#f97316", "#3b82f6", "#22c55e", "#a855f7"]
    _yerr = np.array([[a - lo, hi - a] for a, lo, hi in zip(_ates, _ci_lo, _ci_hi)]).T

    _fig1, _ax1 = plt.subplots(figsize=(10, 5))
    _ax1.bar(
        _labels,
        _ates,
        color=_colors,
        alpha=0.85,
        yerr=_yerr,
        capsize=6,
        error_kw={"linewidth": 1.8, "ecolor": "white", "alpha": 0.7},
    )
    _ax1.axhline(
        1794,
        color="#a855f7",
        linestyle="--",
        linewidth=2,
        label="RCT truth $1,794",
    )
    _ax1.axhline(0, color="white", linestyle="-", linewidth=0.6, alpha=0.35)
    _ax1.set_ylabel("Average Treatment Effect (USD)", color="white", fontsize=11)
    _ax1.set_title(
        "OLS Bias vs Double ML Recovery vs RCT Ground Truth",
        color="white",
        fontsize=13,
        fontweight="bold",
    )
    _ax1.tick_params(colors="white")
    _ax1.set_facecolor("#1e1b4b")
    _fig1.patch.set_facecolor("#1e1b4b")
    for _spine in _ax1.spines.values():
        _spine.set_edgecolor("#ffffff26")
    _ax1.legend(fontsize=10, facecolor="#302b63", labelcolor="white")
    plt.tight_layout()
    st.pyplot(_fig1)
    plt.close(_fig1)

    st.image(str(Path("reports/figures/ate_three_way_table.png")))

with tab2:
    st.subheader("Predict the treatment effect for a custom profile.")
    _col_a, _col_b, _col_c = st.columns(3)
    with _col_a:
        _age = st.slider("Age", 17, 55, 25)
        _education = st.slider("Years of education", 0, 18, 10)
        _re74 = st.slider("re74 (1974 earnings, USD)", 0, 25000, 0, step=100)
    with _col_b:
        _re75 = st.slider("re75 (1975 earnings, USD)", 0, 25000, 0, step=100)
        _black = st.radio("Black", [0, 1], horizontal=True, index=0)
        _hispanic = st.radio("Hispanic", [0, 1], horizontal=True, index=0)
    with _col_c:
        _married = st.radio("Married", [0, 1], horizontal=True, index=0)
        _nodegree = st.radio("No HS diploma", [0, 1], horizontal=True, index=1)

    if st.button("Predict CATE", type="primary"):
        try:
            LalondeProfile(
                age=_age,
                education=_education,
                black=_black,
                hispanic=_hispanic,
                married=_married,
                nodegree=_nodegree,
                re74=float(_re74),
                re75=float(_re75),
            )
        except Exception as _ve:
            st.error(f"Validation error: {_ve}")
            st.stop()

        _X = np.array(
            [
                [
                    _age,
                    _education,
                    _black,
                    _hispanic,
                    _married,
                    _nodegree,
                    float(_re74),
                    float(_re75),
                ]
            ],
            dtype=float,
        )
        _cate_val = float(cf.predict(_X)[0])
        _lo_arr, _hi_arr = cf.estimator.effect_interval(_X, alpha=0.05)
        _ci_lower = float(np.asarray(_lo_arr).reshape(-1)[0])
        _ci_upper = float(np.asarray(_hi_arr).reshape(-1)[0])
        _rec, _nl = translate_cate(_cate_val, _ci_lower, _ci_upper)

        _m1, _m2, _m3 = st.columns(3)
        _m1.metric("Predicted CATE (USD)", f"${_cate_val:,.0f}")
        _m2.metric("95% CI", f"[${_ci_lower:,.0f}, ${_ci_upper:,.0f}]")
        _m3.metric("Recommendation", _rec)
        st.markdown(f"### {_nl}")

        with st.spinner("Computing SHAP moderators…"):
            _shap_vals = explainer.shap_values(
                pd.DataFrame(_X, columns=COVARIATES),
                nsamples=50,
            )
            _expl_obj = shap.Explanation(
                values=np.asarray(_shap_vals),
                base_values=float(explainer.expected_value),
                data=_X,
                feature_names=COVARIATES,
            )
            _fig2, _ax2 = plt.subplots(figsize=(9, 4))
            shap.plots.waterfall(_expl_obj[0], max_display=8, show=False)
            plt.title("Why this CATE? (SHAP moderators)")
            plt.tight_layout()
            st.pyplot(_fig2)
            plt.close(_fig2)

    st.markdown("---")
    st.subheader("Global moderator importance (mean |SHAP|)")
    st.image(str(Path("reports/figures/shap_moderators_bar.png")))
    st.subheader("Per-individual moderator effects (beeswarm)")
    st.image(str(Path("reports/figures/shap_moderators_beeswarm.png")))
    st.markdown(
        """
> **Why "moderator" not "driver"?**
> SHAP on a *predictive* model (e.g. RF predicting re78) identifies what
> drives *outcomes*. SHAP on a *causal forest* (predicting CATE) identifies
> what drives **who benefits more from treatment**. Education has a large
> SHAP value here because it moves the predicted *effect of training*, not
> just baseline earnings. That distinction is the intellectual core of this
> project.
"""
    )

with tab3:
    st.subheader("Tab 3 — Heterogeneity coming soon.")

with tab4:
    st.subheader("Tab 4 — Policy Targeting coming soon.")
