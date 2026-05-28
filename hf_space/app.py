"""Self-contained HF Space app for the Causal ML CATE Explorer (rule C12).

NO `from src.*` imports — Hugging Face Spaces cannot reach the main repo's
`src/` package, so every helper used at runtime (NL translator, theme CSS,
safe prediction guards) is inlined below. The model artefact
``causal_forest.joblib`` is shipped alongside this file via
``huggingface-cli upload hf_space/ .`` and stored on the HF Space side via
LFS (see ``.gitattributes``).

Run locally for smoke test::

    cd hf_space && python app.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import gradio as gr
import joblib
import numpy as np

COVARIATES = [
    "age",
    "education",
    "black",
    "hispanic",
    "married",
    "nodegree",
    "re74",
    "re75",
]

PRIMARY = "#6366f1"
SECONDARY = "#a855f7"

CSS = f"""
@keyframes slideUp {{
  from {{ transform: translateY(20px); opacity: 0; }}
  to   {{ transform: translateY(0);    opacity: 1; }}
}}
.gradio-container {{
  background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #4c1d95 100%);
  min-height: 100vh;
}}
.hero {{
  padding: 24px 28px;
  margin-bottom: 18px;
  background: rgba(255, 255, 255, 0.08);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  border-radius: 16px;
  border: 1px solid rgba(255, 255, 255, 0.12);
  animation: slideUp 0.6s ease-out;
}}
.hero h1 {{
  margin: 0 0 6px 0;
  font-size: 28px;
  font-weight: 700;
  background: linear-gradient(90deg, {PRIMARY}, {SECONDARY});
  -webkit-background-clip: text;
  background-clip: text;
  -webkit-text-fill-color: transparent;
}}
.hero p {{ margin: 4px 0; color: rgba(255, 255, 255, 0.78); font-size: 13px; }}
.hero a {{ color: {SECONDARY}; text-decoration: none; font-weight: 500; }}
.hero a:hover {{ text-decoration: underline; }}
.gr-block, .gr-form, .gr-panel {{
  background: rgba(255, 255, 255, 0.06) !important;
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  border: 1px solid rgba(255, 255, 255, 0.08) !important;
  border-radius: 14px !important;
  animation: slideUp 0.5s ease-out;
}}
button.primary, .gr-button-primary {{
  background: linear-gradient(135deg, {PRIMARY}, {SECONDARY}) !important;
  border: none !important;
  color: white !important;
  font-weight: 600 !important;
}}
"""

_CF = None


def _get_cf() -> object:
    """Lazy-load the EconML CausalForestDML joblib (rule C11)."""
    global _CF
    if _CF is None:
        artefact = Path(__file__).parent / "causal_forest.joblib"
        _CF = joblib.load(str(artefact))  # nosec B301 - shipped artefact
    return _CF


def _safe_predict(cf: object, X: np.ndarray) -> np.ndarray:
    """Inlined NaN/inf/shape guards mirroring src/models/causal_forest_model."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"Expected 2-D X, got shape {X.shape}.")
    if X.shape[1] != len(COVARIATES):
        raise ValueError(f"Expected {len(COVARIATES)} covariates, got {X.shape[1]}.")
    if np.isnan(X).any() or np.isinf(X).any():
        raise ValueError("Input contains NaN or inf.")
    return np.asarray(cf.effect(X)).reshape(-1)  # type: ignore[attr-defined]


def translate_cate(cate: float, ci_lower: float, ci_upper: float) -> tuple[str, str]:
    """Inlined twin of src/api/nl_translator.translate_cate (rule C45)."""
    if ci_lower <= 0 <= ci_upper:
        rec = "DEFER"
        nl = (
            f"No detectable effect for this profile. Predicted CATE is "
            f"${cate:,.0f} but the 95% bootstrap CI [${ci_lower:,.0f}, "
            f"${ci_upper:,.0f}] crosses zero. Recommendation: DEFER."
        )
    elif cate > 0:
        rec = "TREAT"
        nl = (
            f"Job training is predicted to raise this person's 1978 earnings "
            f"by ${cate:,.0f} (95% CI: [${ci_lower:,.0f}, ${ci_upper:,.0f}]). "
            f"Recommendation: TREAT."
        )
    else:
        rec = "DEFER"
        nl = (
            f"Training is predicted to *lower* earnings by ${-cate:,.0f} "
            f"for this profile (95% CI: [${ci_lower:,.0f}, ${ci_upper:,.0f}]). "
            f"Recommendation: DEFER."
        )
    return rec, nl


def stream_cate(
    age: int,
    education: int,
    black: int,
    hispanic: int,
    married: int,
    nodegree: int,
    re74: float,
    re75: float,
) -> Iterator[tuple[str, str, str]]:
    """Stream 5 stage messages while the CATE pipeline runs (rule C43)."""
    yield "Validating input...", "", ""

    X = np.array(
        [[age, education, black, hispanic, married, nodegree, re74, re75]],
        dtype=float,
    )

    yield "Querying causal forest...", "", ""
    cf = _get_cf()
    cate = float(_safe_predict(cf, X)[0])

    yield "Bootstrapping 95% CI...", "", ""
    lo, hi = cf.effect_interval(X, alpha=0.05)  # type: ignore[attr-defined]
    ci_lower = float(np.asarray(lo).reshape(-1)[0])
    ci_upper = float(np.asarray(hi).reshape(-1)[0])

    yield "Translating recommendation...", "", ""
    rec, nl = translate_cate(cate, ci_lower, ci_upper)

    metrics = (
        f"**CATE:** ${cate:,.0f}\n\n"
        f"**95% CI:** [${ci_lower:,.0f}, ${ci_upper:,.0f}]\n\n"
        f"**Recommendation:** {rec}"
    )
    yield "Done.", metrics, nl


def build_demo() -> gr.Blocks:
    """Construct the same gr.Blocks UI as src/api/gradio_demo (mirrored)."""
    repo_url = "https://github.com/Priyrajsinh/causal-ml-hte"
    hero_html = (
        "<div class='hero'>"
        "<h1>Causal ML - Heterogeneous Treatment Effect Explorer</h1>"
        "<p>LaLonde NSW + CPS - Double ML (Chernozhukov 2018) - "
        "CausalForestDML</p>"
        f"<p><a href='{repo_url}'>GitHub repo</a></p>"
        "</div>"
    )
    theme = gr.themes.Base(
        primary_hue=gr.themes.colors.indigo,
        secondary_hue=gr.themes.colors.purple,
        neutral_hue=gr.themes.colors.slate,
    )
    with gr.Blocks(theme=theme, css=CSS, title="Causal ML - HTE CATE Explorer") as demo:
        gr.HTML(hero_html)
        with gr.Row():
            with gr.Column(scale=1):
                age = gr.Slider(17, 55, value=25, step=1, label="Age")
                education = gr.Slider(
                    0, 18, value=10, step=1, label="Years of education"
                )
                black = gr.Radio([0, 1], value=0, label="Black (1 = yes)")
                hispanic = gr.Radio([0, 1], value=0, label="Hispanic")
                married = gr.Radio([0, 1], value=0, label="Married")
                nodegree = gr.Radio([0, 1], value=1, label="No high-school diploma")
                re74 = gr.Slider(
                    0, 25000, value=0, step=100, label="re74 (1974 earnings, USD)"
                )
                re75 = gr.Slider(
                    0, 25000, value=0, step=100, label="re75 (1975 earnings, USD)"
                )
                btn = gr.Button("Predict CATE", variant="primary")
            with gr.Column(scale=2):
                stage_md = gr.Markdown(
                    value="Adjust the profile and click **Predict CATE**."
                )
                metrics_md = gr.Markdown()
                nl_md = gr.Markdown()
        btn.click(
            stream_cate,
            inputs=[
                age,
                education,
                black,
                hispanic,
                married,
                nodegree,
                re74,
                re75,
            ],
            outputs=[stage_md, metrics_md, nl_md],
        )
    return demo


if __name__ == "__main__":
    build_demo().launch()
