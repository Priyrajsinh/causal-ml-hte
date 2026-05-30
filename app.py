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
import pandas as pd
import streamlit as st

from src.config import load_config
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
    st.subheader("Tab 1 — ATE Comparison coming soon.")

with tab2:
    st.subheader("Tab 2 — CATE Explorer coming soon.")

with tab3:
    st.subheader("Tab 3 — Heterogeneity coming soon.")

with tab4:
    st.subheader("Tab 4 — Policy Targeting coming soon.")
