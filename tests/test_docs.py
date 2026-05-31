"""Documentation invariants — guard the recruiter-facing docs against regressions."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(name: str) -> str:
    """Return the text of a doc file at the repository root."""
    with open(str(ROOT / name), encoding="utf-8") as fh:
        return fh.read()


def test_readme_has_live_urls():
    """README must link both live demos (HF Space + Streamlit Cloud)."""
    readme = _read("README.md")
    assert "huggingface.co/spaces/Priyrajsinh/causal-ml-hte" in readme
    assert "streamlit.app" in readme


def test_readme_no_ai_attribution():
    """README + MODEL_CARD must never name an AI assistant or co-author."""
    banned = ["Claude", "Sonnet", "Opus", "Anthropic", "Co-Authored-By"]
    for doc in ("README.md", "MODEL_CARD.md"):
        text = _read(doc)
        for token in banned:
            assert token not in text, f"{token!r} leaked into {doc}"


def test_readme_has_three_way_ate_table():
    """The headline OLS-vs-DML table must label all three data constructions."""
    readme = _read("README.md")
    for label in ("OLS · CPS", "DML · NSW", "RCT ground truth"):
        assert label in readme, f"missing {label!r} in headline ATE table"


def test_readme_has_eu_ai_act_section():
    """README must carry the EU AI Act Annex III §4 framing."""
    readme = _read("README.md")
    assert "Annex III" in readme
    articles = ["Article 9", "Article 10", "Article 13", "Article 14"]
    assert sum(a in readme for a in articles) >= 3


def test_model_card_has_no_TBD():
    """MODEL_CARD must be fully filled — no leftover TBD placeholders."""
    assert "TBD" not in _read("MODEL_CARD.md")


def test_research_notes_has_5_md_files():
    """research-notes/ must contain the 5 numbered reading-log entries."""
    assert len(list((ROOT / "research-notes").glob("0?-*.md"))) == 5
