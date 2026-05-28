"""Self-contained HF Space app for the Causal ML CATE Explorer (rule C12).

NO `from src.*` imports - HF Spaces cannot reach the main repo's `src/`
package, so every helper used at runtime (NL translator, safe-predict
guards) is inlined below. The model artefact ``causal_forest.joblib`` is
shipped alongside this file (see ``.gitattributes``).

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
            f"${ci_upper:,.0f}] crosses zero. Recommendation: **DEFER**."
        )
    elif cate > 0:
        rec = "TREAT"
        nl = (
            f"Job training is predicted to raise this person's 1978 earnings "
            f"by ${cate:,.0f} (95% CI: [${ci_lower:,.0f}, ${ci_upper:,.0f}]). "
            f"Recommendation: **TREAT**."
        )
    else:
        rec = "DEFER"
        nl = (
            f"Training is predicted to *lower* earnings by ${-cate:,.0f} "
            f"for this profile (95% CI: [${ci_lower:,.0f}, ${ci_upper:,.0f}]). "
            f"Recommendation: **DEFER**."
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

    yield "Computing 95% CI...", "", ""
    lo, hi = cf.effect_interval(X, alpha=0.05)  # type: ignore[attr-defined]
    ci_lower = float(np.asarray(lo).reshape(-1)[0])
    ci_upper = float(np.asarray(hi).reshape(-1)[0])

    yield "Translating recommendation...", "", ""
    rec, nl = translate_cate(cate, ci_lower, ci_upper)

    metrics = (
        f"### Result\n\n"
        f"| | |\n|---|---|\n"
        f"| **CATE** | ${cate:,.0f} |\n"
        f"| **95% CI** | [${ci_lower:,.0f}, ${ci_upper:,.0f}] |\n"
        f"| **Recommendation** | **{rec}** |\n"
    )
    yield "Done.", metrics, nl


HEADER_MD = """
# Causal ML - CATE Explorer
LaLonde NSW + CPS - Double ML (Chernozhukov 2018) - CausalForestDML

[GitHub repo](https://github.com/Priyrajsinh/causal-ml-hte)
"""

INTRO_MD = (
    "Adjust the eight pre-treatment covariates on the left, then click "
    "**Predict CATE**. Output appears below."
)


def build_demo() -> gr.Blocks:
    """Construct a minimalist gr.Blocks UI with Soft theme + indigo accent."""
    with gr.Blocks(title="Causal ML - CATE Explorer") as demo:
        gr.Markdown(HEADER_MD)
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Profile")
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
                btn = gr.Button("Predict CATE", variant="primary", size="lg")
            with gr.Column(scale=2):
                gr.Markdown("### Output")
                stage_md = gr.Markdown(value=INTRO_MD)
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
    build_demo().launch(
        theme=gr.themes.Soft(
            primary_hue=gr.themes.colors.indigo,
            secondary_hue=gr.themes.colors.purple,
        )
    )
