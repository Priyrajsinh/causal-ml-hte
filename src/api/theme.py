"""Palette tokens shared across the Gradio (HF) + Streamlit (Day 7) UIs.

Rule C44 pinned the indigo/purple palette; this module exposes the two
brand colors as constants so the Streamlit dashboard can import them
without depending on Gradio.

The Gradio demo (``src/api/gradio_demo.py`` + ``hf_space/app.py``) inlines
its own ``gr.themes.Base(...)`` + CSS - the elaborate theme dance lives
where it is used, not behind a wrapper.
"""

PRIMARY = "#6366f1"
SECONDARY = "#a855f7"
