"""Unit tests for src/api/gradio_demo.py (rule C43 streaming + C11 lazy load)."""

from __future__ import annotations

import inspect

import gradio as gr
import pytest


def test_gradio_demo_does_not_load_model_at_import() -> None:
    """Rule C11: importing src.api.gradio_demo must NOT trigger joblib load."""
    import src.api.gradio_demo as gd

    assert gd._CF is None


def test_stream_predict_is_generator() -> None:
    """``stream_predict`` must be a generator (rule C43 streaming pattern)."""
    from src.api.gradio_demo import stream_predict

    assert inspect.isgeneratorfunction(stream_predict)


def test_build_demo_returns_blocks() -> None:
    """``build_demo`` returns a gr.Blocks instance ready to launch."""
    from src.api.gradio_demo import build_demo

    demo = build_demo()
    assert isinstance(demo, gr.Blocks)


def test_examples_match_input_arity() -> None:
    """Each example must have 8 entries (one per covariate)."""
    from src.api.gradio_demo import EXAMPLE_LABELS, EXAMPLES

    assert len(EXAMPLES) == len(EXAMPLE_LABELS)
    for row in EXAMPLES:
        assert len(row) == 8


def test_decision_card_renders_treat_with_green() -> None:
    """``_decision_card`` emits the green ``decision-treat`` class for TREAT."""
    from src.api.gradio_demo import _decision_card

    html = _decision_card("TREAT", 2000.0, 500.0, 3500.0, "test")
    assert "decision-treat" in html
    assert "RECOMMEND TRAINING" in html
    assert "$2,000" in html


def test_decision_card_renders_defer_with_amber_when_ci_crosses_zero() -> None:
    """CI crossing zero -> amber INSUFFICIENT EVIDENCE card."""
    from src.api.gradio_demo import _decision_card

    html = _decision_card("DEFER", 200.0, -100.0, 500.0, "test")
    assert "decision-defer" in html
    assert "INSUFFICIENT EVIDENCE" in html


def test_decision_card_renders_warning_when_ci_below_zero() -> None:
    """CI entirely below zero -> red DO NOT TREAT card."""
    from src.api.gradio_demo import _decision_card

    html = _decision_card("DEFER", -500.0, -1000.0, -100.0, "test")
    assert "decision-warning" in html
    assert "DO NOT TREAT" in html


@pytest.mark.slow
def test_stream_predict_yields_pipeline_and_card_when_model_present() -> None:
    """``stream_predict`` yields (pipeline_html, result_html) tuples.

    Final yield carries the decision card. Requires the causal forest
    joblib - marked slow + skip-if-missing.
    """
    from pathlib import Path

    if not Path("models/causal_forest/causal_forest.joblib").exists():
        pytest.skip("causal_forest.joblib not materialised")
    from src.api.gradio_demo import stream_predict

    gen = stream_predict(25, 10, 1, 0, 0, 1, 0.0, 0.0)
    items = list(gen)
    # 4 stage yields + 1 final-with-card = 5
    assert len(items) == 5
    for pipeline, card in items:
        assert isinstance(pipeline, str)
        assert isinstance(card, str)
    # Final yield has the decision card
    final_pipeline, final_card = items[-1]
    assert "Done" in final_pipeline
    assert final_card != ""
    assert "Predicted effect" in final_card
    # Should land on one of the three decision classes
    assert any(
        cls in final_card
        for cls in ("decision-treat", "decision-defer", "decision-warning")
    )
