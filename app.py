"""Streamlit 4-tab dashboard — Causal ML · HTE (Day 7).

Tabs:
  1 · ATE Comparison  — OLS-bias vs DML-recovery vs RCT ground truth
  2 · CATE Explorer   — live Gradio Space embedded (model runs on HF)
  3 · Heterogeneity   — who benefits more? quartile bars + scatters
  4 · Policy Targeting — top-30% policy curve + lift estimate

Model artefacts (1.9 GB) are not committed to git.  Tab 2 embeds the
live HF Gradio Space so predictions work without loading the model here.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src.config import load_config

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
    """Load config + results JSON.  No model files required."""
    cfg = load_config("config/config.yaml")
    results = json.loads(Path(cfg["paths"]["results_json"]).read_text())
    return cfg, results


cfg, results = load_artifacts()

# ── Hero ──────────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="hero">
  <h1>Causal ML · Heterogeneous Treatment Effects</h1>
  <p>LaLonde NSW + CPS &nbsp;·&nbsp; Double ML (Chernozhukov 2018)
     &nbsp;·&nbsp; CausalForestDML &nbsp;·&nbsp; SHAP moderators</p>
  <div class="hero-links">
    <a href="https://huggingface.co/spaces/Priyrajsinh/causal-ml-hte-cate-explorer"
       target="_blank">Live CATE Predictor (Gradio)</a>
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

# ── Tab 1: ATE Comparison ─────────────────────────────────────────────────
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

# ── Tab 2: CATE Explorer (embedded HF Gradio Space) ──────────────────────
with tab2:
    st.subheader("Predict the treatment effect for a custom profile.")
    st.markdown(
        "Enter a profile using the sliders below and click **Predict CATE**. "
        "The model runs on the Hugging Face Space — "
        "the first load may take ~20 seconds if the Space is waking up."
    )
    components.iframe(
        "https://priyrajsinh-causal-ml-hte-cate-explorer.hf.space",
        height=860,
        scrolling=True,
    )
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

# ── Tab 3: Heterogeneity ──────────────────────────────────────────────────
with tab3:
    st.subheader("Who benefits more from job training?")
    _c1, _c2, _c3 = st.columns(3)
    with _c1:
        st.image(str(Path("reports/figures/heterogeneity_education.png")))
    with _c2:
        st.image(str(Path("reports/figures/heterogeneity_age.png")))
    with _c3:
        st.image(str(Path("reports/figures/heterogeneity_re74.png")))

    st.markdown("---")
    st.subheader("Continuous patterns (rolling mean overlay)")
    _c1b, _c2b, _c3b = st.columns(3)
    with _c1b:
        st.image(str(Path("reports/figures/heterogeneity_education_scatter.png")))
    with _c2b:
        st.image(str(Path("reports/figures/heterogeneity_age_scatter.png")))
    with _c3b:
        st.image(str(Path("reports/figures/heterogeneity_re74_scatter.png")))

    st.markdown("---")
    st.subheader("Moderator-score ranking (CATE range per covariate)")
    _scores = pd.read_csv("reports/figures/moderator_scores.csv")
    st.dataframe(
        _scores.style.format(
            {
                "cate_range": "${:,.0f}",
                "cate_min": "${:,.0f}",
                "cate_max": "${:,.0f}",
            }
        ),
        use_container_width=True,
    )

# ── Tab 4: Policy Targeting ───────────────────────────────────────────────
with tab4:
    st.subheader("Top-30% targeting by predicted CATE")
    _policy = results["policy_targeting"]
    _p1, _p2, _p3, _p4 = st.columns(4)
    _p1.metric(
        "Targeted (n)",
        f"{_policy['n_targeted']} / {_policy['n_total']}",
    )
    _p2.metric("Avg CATE — targeted", f"${_policy['mean_cate_targeted']:,.0f}")
    _p3.metric(
        "Avg CATE — untargeted",
        f"${_policy['mean_cate_untargeted']:,.0f}",
    )
    _p4.metric(
        "Total realised lift (USD)",
        f"${_policy['estimated_total_lift_usd']:,.0f}",
    )

    st.image(str(Path("reports/figures/policy_curve.png")))
    st.markdown(
        """
> The **policy curve** shows how much total earnings lift is captured if we
> treat the top-K% of applicants ranked by predicted CATE. The targeted
> curve climbing above the random baseline is empirical evidence that CATE
> predictions carry signal — targeting based on the model is materially
> better than first-come-first-served.
"""
    )
    st.image(str(Path("reports/figures/policy_top30_profile.png")))
    st.markdown(
        "**Who would be targeted?** The bar chart above shows how the average "
        "covariate values of the targeted group differ from the population mean. "
        "Positive bars = the targeted group scores higher than average on that "
        "covariate."
    )
