"""Tests for the top-level Streamlit dashboard (app.py).

Non-slow: test_glass_css_exists — runs in every CI pass, no model needed.
Slow: AppTest-based — skipped in CI (addopts = '-m not slow') but runnable
      locally.  No model files required since load_artifacts() only reads
      the committed results.json.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_APP_PATH = "app.py"


def test_glass_css_exists() -> None:
    """The glassmorphism CSS file must exist for the Streamlit app to load."""
    assert Path("src/api/streamlit_glass.css").exists()


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_app_module_imports() -> None:
    """app.py loads in an AppTest context without raising an exception."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(_APP_PATH, default_timeout=60)
    at.run()
    assert not at.exception, str(at.exception)


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_hero_renders() -> None:
    """Hero section contains the 'Causal ML' heading."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(_APP_PATH, default_timeout=60)
    at.run()
    assert not at.exception, str(at.exception)
    all_md = " ".join(md.value for md in at.markdown)
    assert "Causal ML" in all_md


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_four_tabs_present() -> None:
    """All four tab sections render a subheader."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(_APP_PATH, default_timeout=60)
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
def test_tab2_iframe_renders() -> None:
    """Tab 2 renders the HF Space iframe without exception."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(_APP_PATH, default_timeout=60)
    at.run()
    assert not at.exception, str(at.exception)
    # Tab 2 subheader confirms the section rendered
    headings = [s.value for s in at.subheader]
    assert any("custom profile" in h.lower() for h in headings)
