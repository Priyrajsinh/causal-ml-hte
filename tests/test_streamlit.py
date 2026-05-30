"""Tests for the top-level Streamlit dashboard (app.py, Day 7).

Non-slow tests: test_glass_css_exists only (runs in every CI pass).
Slow tests: AppTest-based, require the fitted causal forest model.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_MODEL_PATH = Path("models/causal_forest/causal_forest.joblib")
_APP_PATH = "app.py"


def test_glass_css_exists() -> None:
    """The glassmorphism CSS file must exist for the Streamlit app to load."""
    assert Path("src/api/streamlit_glass.css").exists()


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_app_module_imports() -> None:
    """app.py loads in an AppTest context without raising an exception."""
    if not _MODEL_PATH.exists():
        pytest.skip("causal_forest.joblib not materialised")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(_APP_PATH, default_timeout=120)
    at.run()
    assert not at.exception, str(at.exception)


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_hero_renders() -> None:
    """Hero section contains the 'Causal ML' heading."""
    if not _MODEL_PATH.exists():
        pytest.skip("causal_forest.joblib not materialised")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(_APP_PATH, default_timeout=120)
    at.run()
    assert not at.exception, str(at.exception)
    all_md = " ".join(md.value for md in at.markdown)
    assert "Causal ML" in all_md


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_four_tabs_present() -> None:
    """All four tab sections are rendered (one subheader per tab)."""
    if not _MODEL_PATH.exists():
        pytest.skip("causal_forest.joblib not materialised")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(_APP_PATH, default_timeout=120)
    at.run()
    assert not at.exception, str(at.exception)
    headings = [s.value for s in at.subheader]
    assert any(
        "OLS" in h or "Naive" in h or "Double ML" in h for h in headings
    ), "Tab 1 subheader missing"
    assert any(
        "treatment effect" in h.lower() or "custom profile" in h.lower()
        for h in headings
    ), "Tab 2 subheader missing"
    assert any(
        "benefit" in h.lower() or "heterogeneity" in h.lower() for h in headings
    ), "Tab 3 subheader missing"
    assert any(
        "targeting" in h.lower() or "top-30" in h.lower() for h in headings
    ), "Tab 4 subheader missing"


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_predict_button_in_tab2() -> None:
    """Clicking 'Predict CATE' completes without raising an exception."""
    if not _MODEL_PATH.exists():
        pytest.skip("causal_forest.joblib not materialised")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(_APP_PATH, default_timeout=300)
    at.run()
    assert not at.exception, str(at.exception)
    at.button[0].click()
    at.run()
    assert not at.exception, str(at.exception)
