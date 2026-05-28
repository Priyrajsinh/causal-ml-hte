"""Unit tests for src/api/nl_translator.py (rule C45)."""

from src.api.nl_translator import translate_cate


def test_translate_treat_when_ci_above_zero() -> None:
    """Positive CATE with CI entirely above zero -> TREAT + raise wording."""
    rec, nl = translate_cate(cate=2000.0, ci_lower=500.0, ci_upper=3500.0)
    assert rec == "TREAT"
    assert "raise" in nl
    assert "$2,000" in nl
    assert "$500" in nl
    assert "$3,500" in nl


def test_translate_defer_when_ci_crosses_zero() -> None:
    """CI crosses zero -> DEFER + no-detectable-effect wording (rule C45)."""
    rec, nl = translate_cate(cate=200.0, ci_lower=-100.0, ci_upper=500.0)
    assert rec == "DEFER"
    assert "No detectable effect" in nl
    assert "crosses zero" in nl


def test_translate_defer_when_negative() -> None:
    """Negative CATE with CI entirely below zero -> DEFER + *lower* wording."""
    rec, nl = translate_cate(cate=-500.0, ci_lower=-1000.0, ci_upper=-100.0)
    assert rec == "DEFER"
    assert "*lower*" in nl
    assert "$500" in nl


def test_translate_defer_when_ci_just_touches_zero_at_lower() -> None:
    """Boundary case: ci_lower == 0 with ci_upper > 0 still crosses zero -> DEFER."""
    rec, nl = translate_cate(cate=100.0, ci_lower=0.0, ci_upper=500.0)
    assert rec == "DEFER"
    assert "No detectable effect" in nl


def test_translate_treat_strictly_positive_ci() -> None:
    """Strictly positive CI (lower bound > 0) -> TREAT."""
    rec, _ = translate_cate(cate=1794.0, ci_lower=550.0, ci_upper=3038.0)
    assert rec == "TREAT"
