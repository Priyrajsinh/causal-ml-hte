"""Streaming Gradio CATE explorer (rule C43).

Runs locally via ``make gradio`` or ``python -m src.api.gradio_demo``. The
HF Space deploy uses a frozen, self-contained copy at ``hf_space/app.py``
(rule C12 - Spaces cannot import from ``src/``).

Streaming pattern (rule C43): ``stream_cate`` is a generator that yields
five stage messages as the prediction pipeline progresses (validate,
forest, CI, translate, done). Each yield refreshes the UI so the user
sees the inference proceeding rather than a frozen spinner.
"""

from pathlib import Path
from typing import Iterator

import gradio as gr
import numpy as np

from src.api.nl_translator import translate_cate
from src.api.theme import get_theme
from src.config import load_config
from src.models.causal_forest_model import CausalForestEstimator

_CF: CausalForestEstimator | None = None


def _get_cf() -> CausalForestEstimator:
    """Lazy-load the causal forest on first call (rule C11: no import-time I/O)."""
    global _CF
    if _CF is None:
        cfg = load_config("config/config.yaml")
        _CF = CausalForestEstimator.load(
            Path(cfg["paths"]["models_dir"]) / "causal_forest"
        )
    return _CF


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
    """Yield ``(stage_md, metrics_md, nl_md)`` 5 times as the pipeline progresses."""
    yield "Validating input...", "", ""

    X = np.array(
        [[age, education, black, hispanic, married, nodegree, re74, re75]],
        dtype=float,
    )

    yield "Querying causal forest...", "", ""
    cf = _get_cf()
    cate = float(cf.safe_predict(X)[0])

    yield "Computing 95% CI...", "", ""
    assert cf.estimator is not None
    lo, hi = cf.estimator.effect_interval(X, alpha=0.05)
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

[GitHub repo](https://github.com/Priyrajsinh/causal-ml-hte) -
[HF Space](https://huggingface.co/spaces/Priyrajsinh/causal-ml-hte-cate-explorer)
"""

INTRO_MD = (
    "Adjust the eight pre-treatment covariates on the left, then click "
    "**Predict CATE**. Output appears below."
)


def build_demo() -> gr.Blocks:
    """Construct the gr.Blocks UI with the Soft theme."""
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


def main() -> None:
    """Launch the demo on 0.0.0.0:7860."""
    build_demo().launch(
        server_name="0.0.0.0",  # nosec B104
        server_port=7860,
        theme=get_theme(),
    )


if __name__ == "__main__":
    main()
