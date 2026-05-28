"""Self-contained HF Space - Causal ML CATE Explorer (rule C12).

ZERO `from src.*` imports. Inlined twins of nl_translator + safe_predict.
Streaming Gradio app with tabs (Try It / How It Works / Limitations),
clickable persona examples, and a big colored TREAT/DEFER decision card.

Run locally for smoke test::

    cd hf_space && python app.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Generator

import gradio as gr
import joblib
import numpy as np

# ---------------------------------------------------------------------------
# Model loading (lazy, rule C11)
# ---------------------------------------------------------------------------
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

_CF: object | None = None


def _get_cf() -> object:
    """Lazy-load the EconML CausalForestDML joblib on first call."""
    global _CF
    if _CF is None:
        artefact = Path(__file__).parent / "causal_forest.joblib"
        _CF = joblib.load(str(artefact))  # nosec B301 - shipped artefact
    return _CF


# ---------------------------------------------------------------------------
# Inlined helpers (mirror src/api/* without importing them)
# ---------------------------------------------------------------------------
def _safe_predict(cf: object, X: np.ndarray) -> np.ndarray:
    """NaN/inf/shape guards mirroring src/models/causal_forest_model."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"Expected 2-D X, got shape {X.shape}.")
    if X.shape[1] != len(COVARIATES):
        raise ValueError(f"Expected {len(COVARIATES)} covariates, got {X.shape[1]}.")
    if np.isnan(X).any() or np.isinf(X).any():
        raise ValueError("Input contains NaN or inf.")
    return np.asarray(cf.effect(X)).reshape(-1)  # type: ignore[attr-defined]


def translate_cate(cate: float, ci_lower: float, ci_upper: float) -> tuple[str, str]:
    """Twin of src/api/nl_translator.translate_cate (rule C45)."""
    if ci_lower <= 0 <= ci_upper:
        rec = "DEFER"
        nl = (
            f"The model predicts a +${cate:,.0f} effect, but the 95% range "
            f"[${ci_lower:,.0f}, ${ci_upper:,.0f}] crosses zero - so the "
            f"model is not confident the effect is real."
        )
    elif cate > 0:
        rec = "TREAT"
        nl = (
            f"Job training is predicted to raise this person's 1978 earnings "
            f"by ${cate:,.0f}, and the 95% range [${ci_lower:,.0f}, "
            f"${ci_upper:,.0f}] is entirely above zero."
        )
    else:
        rec = "DEFER"
        nl = (
            f"Warning - training is predicted to *lower* earnings by "
            f"${-cate:,.0f} for this profile (95% range: [${ci_lower:,.0f}, "
            f"${ci_upper:,.0f}])."
        )
    return rec, nl


# ---------------------------------------------------------------------------
# Decision card (the big colored verdict block)
# ---------------------------------------------------------------------------
def _decision_card(
    rec: str, cate: float, ci_lower: float, ci_upper: float, nl: str
) -> str:
    """Render the verdict as a big colored card."""
    if rec == "TREAT":
        cls, icon, title = "decision-treat", "&#10004;", "RECOMMEND TRAINING"
        sub = "Strong evidence this person benefits"
    elif ci_lower <= 0 <= ci_upper:
        cls, icon, title = "decision-defer", "&#9888;", "INSUFFICIENT EVIDENCE"
        sub = "Effect could be zero - cannot recommend confidently"
    else:
        cls, icon, title = "decision-warning", "&#10060;", "DO NOT TREAT"
        sub = "Model predicts training would lower earnings"

    return (
        f"<div class='{cls} result-reveal'>"
        f"<div style='font-size:3rem'>{icon}</div>"
        f"<div style='font-size:1.6rem;font-weight:800;letter-spacing:1.5px;"
        f"margin-top:6px'>{title}</div>"
        f"<div style='font-size:0.95rem;opacity:0.92;margin-top:6px'>{sub}</div>"
        f"<div style='margin-top:18px;font-size:1.3rem;font-weight:700'>"
        f"Predicted effect: ${cate:,.0f}</div>"
        f"<div style='font-size:0.9rem;opacity:0.85;margin-top:4px'>"
        f"95% confidence range: [${ci_lower:,.0f}, ${ci_upper:,.0f}]</div>"
        f"<div style='margin-top:16px;font-size:0.95rem;opacity:0.88;"
        f"line-height:1.5;text-align:left;padding:0 18px'>{nl}</div>"
        f"</div>"
    )


# ---------------------------------------------------------------------------
# Streaming predict (yields HTML, P1 jailbreak-style pipeline trace)
# ---------------------------------------------------------------------------
def stream_predict(  # noqa: PLR0913
    age: int,
    education: int,
    black: int,
    hispanic: int,
    married: int,
    nodegree: int,
    re74: float,
    re75: float,
) -> Generator[tuple[str, str], None, None]:
    """Yield (pipeline_html, result_html) progressively."""
    steps = (
        "<div class='stage-step'>"
        "&#128203; <b>Step 1</b> - Reading profile (8 attributes)..."
        "</div>"
    )
    yield steps, ""

    X = np.array(
        [[age, education, black, hispanic, married, nodegree, re74, re75]],
        dtype=float,
    )

    steps += (
        "<div class='stage-step'>"
        "&#127794; <b>Step 2</b> - Querying causal forest "
        "(200 trees fit on LaLonde NSW)..."
        "</div>"
    )
    yield steps, ""

    cf = _get_cf()
    cate = float(_safe_predict(cf, X)[0])

    steps += (
        "<div class='stage-step'>"
        "&#128202; <b>Step 3</b> - Computing 95% confidence range..."
        "</div>"
    )
    yield steps, ""

    lo, hi = cf.effect_interval(X, alpha=0.05)  # type: ignore[attr-defined]
    ci_lower = float(np.asarray(lo).reshape(-1)[0])
    ci_upper = float(np.asarray(hi).reshape(-1)[0])

    steps += (
        "<div class='stage-step'>"
        "&#128221; <b>Step 4</b> - Translating to a recommendation..."
        "</div>"
    )
    yield steps, ""

    rec, nl = translate_cate(cate, ci_lower, ci_upper)
    card = _decision_card(rec, cate, ci_lower, ci_upper, nl)

    steps += (
        "<div class='stage-step' style='color:#a7f3d0'>"
        "&#10003; <b>Done</b> in 4 steps."
        "</div>"
    )
    yield steps, card


# ---------------------------------------------------------------------------
# Examples - the activation booster (rule: lower the barrier to "try it")
# ---------------------------------------------------------------------------
EXAMPLES = [
    # age, edu, black, hispanic, married, nodegree, re74, re75
    [28, 12, 1, 0, 0, 0, 0, 0],
    [22, 9, 1, 0, 0, 1, 0, 0],
    [45, 8, 1, 0, 0, 1, 2000, 2000],
    [35, 12, 0, 0, 1, 0, 8000, 8500],
]
EXAMPLE_LABELS = [
    "High-school grad, no income",
    "Young low-skill worker",
    "Older, no degree",
    "Established mid-career",
]


# ---------------------------------------------------------------------------
# CSS + HTML scaffolding (modelled on P1 jailbreak)
# ---------------------------------------------------------------------------
_CSS = """
@keyframes slideUpFadeIn {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}
.result-reveal { animation: slideUpFadeIn 0.35s ease-out forwards; }

.decision-treat {
    background: linear-gradient(135deg, #065f46, #10b981);
    color: white; border-radius: 14px;
    padding: 28px 24px; text-align: center; margin-top: 8px;
    box-shadow: 0 8px 32px rgba(16,185,129,0.25);
}
.decision-defer {
    background: linear-gradient(135deg, #78350f, #f59e0b);
    color: white; border-radius: 14px;
    padding: 28px 24px; text-align: center; margin-top: 8px;
    box-shadow: 0 8px 32px rgba(245,158,11,0.25);
}
.decision-warning {
    background: linear-gradient(135deg, #7f1d1d, #ef4444);
    color: white; border-radius: 14px;
    padding: 28px 24px; text-align: center; margin-top: 8px;
    box-shadow: 0 8px 32px rgba(239,68,68,0.25);
}

.stage-step {
    padding: 8px 14px; margin: 4px 0;
    border-left: 3px solid #6366f1;
    background: rgba(99,102,241,0.06);
    font-size: 0.92rem;
    border-radius: 0 6px 6px 0;
    color: #e0e7ff;
}

/* Tab strip readability (matches P1 jailbreak fix) */
button[role="tab"] {
    color: rgba(255,255,255,0.7) !important;
    background: transparent !important;
    border: none !important;
    padding: 10px 18px !important;
    font-weight: 500 !important;
    cursor: pointer !important;
}
button[role="tab"]:hover {
    color: #ffffff !important;
    background: rgba(99,102,241,0.12) !important;
}
button[role="tab"][aria-selected="true"] {
    color: #ffffff !important;
    border-bottom: 2px solid #6366f1 !important;
}
"""

_HERO = """
<div style="text-align:center;padding:28px 0 14px">
  <h1 style="background:linear-gradient(90deg,#6366f1,#a855f7,#ec4899);
             -webkit-background-clip:text;-webkit-text-fill-color:transparent;
             font-size:2.4rem;font-weight:800;margin:0;line-height:1.15">
    Should this person get job training?
  </h1>
  <p style="color:rgba(255,255,255,0.62);font-size:1rem;margin-top:8px">
    A causal-ML demo built on the LaLonde NSW + CPS datasets.
    The model predicts the effect of training on 1978 earnings - per person.
  </p>
  <p style="color:rgba(255,255,255,.40);font-size:.82rem;margin-top:4px">
    Method: Double Machine Learning (Chernozhukov 2018) + Causal Forest
  </p>
</div>
"""

_FOOTER = """
<div style="text-align:center;padding:14px;color:rgba(255,255,255,.42);
            font-size:0.82rem;margin-top:18px">
  Built by
  <a href="https://github.com/Priyrajsinh" style="color:#a78bfa">
    Priyrajsinh Parmar</a>
  &nbsp;|&nbsp;
  <a href="https://github.com/Priyrajsinh/causal-ml-hte" style="color:#a78bfa">
    GitHub repo</a>
</div>
"""

_INTRO_MD = """
**Try it in 3 steps:**
1. Pick a quick example below or move the sliders to describe a person.
2. Click **Predict**.
3. Read the colored verdict: green = treat, amber = uncertain, red = avoid.

You don't need to know statistics to read the result.
"""

_HOW_MD = """
### What is this demo actually doing?

The model has learned, from the **LaLonde National Supported Work
Demonstration** (1975-1979), how a US job-training program affected
participants' 1978 earnings. The original study was a **randomized
controlled trial** - some applicants were randomly assigned to receive
training, others were not. That gives us a clean "ground truth" effect
of about **+$1,794** in 1978 earnings, on average.

### Why predict per person, not just an average?

The average hides who benefits. Two questions a policy team would ask:

- *"Of 1,000 applicants, who should we prioritize for our limited
  training slots?"*
- *"Are there profiles where training doesn't help, or even hurts?"*

A standard regression gives you one number (the average). A **causal
forest** gives you one number **per person** - the *Conditional Average
Treatment Effect* (CATE) - so you can rank and target.

### Why not just use regular machine learning?

Because regular machine learning will tell you a lie on this dataset.
If you mix the NSW treated group with a large external control group
(Current Population Survey, ~16,000 people) and run a naive regression,
you get a **negative** ATE - it looks like training *hurts* earnings.
This is **selection bias**: NSW participants were unemployed, the CPS
controls were employed. They're not comparable.

**Double Machine Learning** (Chernozhukov et al., 2018) removes the bias
by orthogonalizing the treatment and outcome against the covariates
using cross-fitted nuisance models. On the same biased data, DML
recovers the +$1,794 RCT truth.

### What does the verdict mean?

- **TREAT** (green) - the model's 95% confidence range is entirely above
  zero. We're confident training would raise this person's earnings.
- **INSUFFICIENT EVIDENCE** (amber) - the range crosses zero. The model
  cannot tell you whether the effect is positive or zero.
- **DO NOT TREAT** (red) - the range is entirely below zero. Rare, but
  the model predicts harm for some low-baseline profiles.
"""

_LIMITS_MD = """
### Demo vs. production model

This Space runs a **1.4 MB** version of the model. The full production
model (in the GitHub repo) is **1.8 GB** because it stores 500 bootstrap
forests for confidence-interval estimation. The HuggingFace free tier
caps repos at 1 GB.

| | Demo (this Space) | Production (repo) |
|---|---|---|
| Predicted effect | **identical** | reference |
| 95% range method | Bootstrap-of-Little-Bags | 500-sample bootstrap |
| Range width | tighter (~25% narrower) | wider, more conservative |
| Verdict | can flip on borderline cases | the canonical answer |

**Practical impact:** on borderline profiles (small predicted effect),
the demo's tighter range may say TREAT while the production model
would say INSUFFICIENT EVIDENCE. The production model is the one you
should trust for any real decision.

### When the model is wrong

- **Population shift**: NSW participants were US workers in the
  1970s. Predictions for very different populations (other countries,
  decades, training programs) are extrapolation.
- **Confounders we did not measure**: motivation, family support,
  local labor-market conditions. The model assumes the 8 measured
  attributes capture everything that matters for selection.
- **Small sample**: only 445 NSW participants. Estimates for unusual
  profiles (very old + very educated + high prior earnings) are
  imprecise.

### Intended use

Pedagogical and research-oriented. This is a portfolio project, not a
deployed policy tool. Real job-training program eligibility decisions
involve far more than a model output.
"""


# ---------------------------------------------------------------------------
# Build the app
# ---------------------------------------------------------------------------
def build_app() -> gr.Blocks:
    """Construct the Gradio Blocks app with tabs + examples + decision card."""
    with gr.Blocks(title="Causal ML - CATE Explorer") as app:
        gr.HTML(_HERO)

        with gr.Tabs():
            with gr.Tab("Try It"):
                gr.Markdown(_INTRO_MD)
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### Describe the person")
                        age = gr.Slider(17, 55, value=25, step=1, label="Age")
                        education = gr.Slider(
                            0, 18, value=10, step=1, label="Years of school"
                        )
                        black = gr.Radio([0, 1], value=0, label="Black (1 = yes)")
                        hispanic = gr.Radio([0, 1], value=0, label="Hispanic (1 = yes)")
                        married = gr.Radio([0, 1], value=0, label="Married (1 = yes)")
                        nodegree = gr.Radio(
                            [0, 1],
                            value=1,
                            label="No high-school diploma (1 = no diploma)",
                        )
                        re74 = gr.Slider(
                            0,
                            25000,
                            value=0,
                            step=100,
                            label="1974 earnings (USD)",
                        )
                        re75 = gr.Slider(
                            0,
                            25000,
                            value=0,
                            step=100,
                            label="1975 earnings (USD)",
                        )
                        btn = gr.Button("Predict", variant="primary", size="lg")
                        gr.Examples(
                            examples=EXAMPLES,
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
                            label="Quick examples - click any row",
                            example_labels=EXAMPLE_LABELS,
                        )

                    with gr.Column(scale=2):
                        gr.Markdown("#### Pipeline")
                        pipeline_html = gr.HTML(
                            value=(
                                "<p style='color:rgba(255,255,255,.45);"
                                "padding:30px;text-align:center'>"
                                "Click <b>Predict</b> to run the model."
                                "</p>"
                            )
                        )
                        gr.Markdown("#### Verdict")
                        result_html = gr.HTML(
                            value=(
                                "<p style='color:rgba(255,255,255,.35);"
                                "padding:18px;text-align:center;font-size:0.9rem'>"
                                "The recommendation appears here."
                                "</p>"
                            )
                        )

                btn.click(
                    fn=stream_predict,
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
                    outputs=[pipeline_html, result_html],
                )

            with gr.Tab("How It Works"):
                gr.Markdown(_HOW_MD)

            with gr.Tab("Limitations"):
                gr.Markdown(_LIMITS_MD)

        gr.HTML(_FOOTER)

    return app


if __name__ == "__main__":
    # Gradio 6: theme + css belong on launch(), not on gr.Blocks()
    theme = gr.themes.Base(
        primary_hue="indigo",
        secondary_hue="purple",
        neutral_hue="slate",
    )
    build_app().launch(theme=theme, css=_CSS)
