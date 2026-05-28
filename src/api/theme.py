"""Glassmorphism theme + CSS for the Gradio CATE explorer (rule C44).

Palette pinned to config.yaml -> ui:
- primary  #6366f1 (indigo-500)
- secondary #a855f7 (purple-500)

The same CSS string is consumed by ``src/api/gradio_demo.py`` and by the
self-contained ``hf_space/app.py`` (which inlines a frozen copy so it can
satisfy rule C12: NO ``from src.*`` imports inside the HF Space).
"""

import gradio as gr

PRIMARY = "#6366f1"
SECONDARY = "#a855f7"

CSS = f"""
@keyframes slideUp {{
  from {{ transform: translateY(20px); opacity: 0; }}
  to   {{ transform: translateY(0);    opacity: 1; }}
}}

.gradio-container {{
  background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #4c1d95 100%);
  min-height: 100vh;
}}

.hero {{
  padding: 24px 28px;
  margin-bottom: 18px;
  background: rgba(255, 255, 255, 0.08);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  border-radius: 16px;
  border: 1px solid rgba(255, 255, 255, 0.12);
  animation: slideUp 0.6s ease-out;
}}

.hero h1 {{
  margin: 0 0 6px 0;
  font-size: 28px;
  font-weight: 700;
  background: linear-gradient(90deg, {PRIMARY}, {SECONDARY});
  -webkit-background-clip: text;
  background-clip: text;
  -webkit-text-fill-color: transparent;
}}

.hero p {{
  margin: 4px 0;
  color: rgba(255, 255, 255, 0.78);
  font-size: 13px;
}}

.hero a {{
  color: {SECONDARY};
  text-decoration: none;
  font-weight: 500;
}}

.hero a:hover {{ text-decoration: underline; }}

.gr-block, .gr-form, .gr-panel {{
  background: rgba(255, 255, 255, 0.06) !important;
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  border: 1px solid rgba(255, 255, 255, 0.08) !important;
  border-radius: 14px !important;
  animation: slideUp 0.5s ease-out;
}}

button.primary, .gr-button-primary {{
  background: linear-gradient(135deg, {PRIMARY}, {SECONDARY}) !important;
  border: none !important;
  color: white !important;
  font-weight: 600 !important;
}}
"""


def get_css() -> str:
    """Return the project's glassmorphism CSS string (rule C44)."""
    return CSS


def get_theme() -> gr.themes.Base:
    """Return a Gradio theme tuned to the indigo/purple palette."""
    return gr.themes.Base(
        primary_hue=gr.themes.colors.indigo,
        secondary_hue=gr.themes.colors.purple,
        neutral_hue=gr.themes.colors.slate,
    )
