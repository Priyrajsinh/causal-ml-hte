"""Unit tests for src/api/gradio_demo.py (rule C43 streaming + C11 lazy load)."""

from __future__ import annotations

import inspect

import gradio as gr
import pytest


def test_theme_exposes_indigo_purple_palette() -> None:
    """``get_theme`` returns the project's tuned Soft theme."""
    from src.api.theme import PRIMARY, SECONDARY, get_theme

    assert PRIMARY == "#6366f1"
    assert SECONDARY == "#a855f7"
    theme = get_theme()
    assert isinstance(theme, gr.themes.Soft)


def test_gradio_demo_does_not_load_model_at_import() -> None:
    """Rule C11: importing src.api.gradio_demo must NOT trigger joblib load."""
    import src.api.gradio_demo as gd

    assert gd._CF is None


def test_stream_cate_is_generator() -> None:
    """``stream_cate`` must be a generator (rule C43 streaming pattern)."""
    from src.api.gradio_demo import stream_cate

    assert inspect.isgeneratorfunction(stream_cate)


def test_build_demo_returns_blocks() -> None:
    """``build_demo`` returns a gr.Blocks instance ready to launch."""
    from src.api.gradio_demo import build_demo

    demo = build_demo()
    assert isinstance(demo, gr.Blocks)


@pytest.mark.slow
def test_stream_cate_yields_five_stages_when_model_present() -> None:
    """``stream_cate`` yields exactly 5 tuples (stage, metrics, nl).

    Requires the causal forest joblib - marked slow + skip-if-missing.
    """
    from pathlib import Path

    if not Path("models/causal_forest/causal_forest.joblib").exists():
        pytest.skip("causal_forest.joblib not materialised")
    from src.api.gradio_demo import stream_cate

    gen = stream_cate(25, 10, 1, 0, 0, 1, 0.0, 0.0)
    items = list(gen)
    assert len(items) == 5
    for stage, metrics, nl in items:
        assert isinstance(stage, str)
        assert isinstance(metrics, str)
        assert isinstance(nl, str)
    # Last yield carries the final result
    final_stage, final_metrics, final_nl = items[-1]
    assert "Done" in final_stage
    assert "CATE" in final_metrics
    assert "Recommendation" in final_metrics
