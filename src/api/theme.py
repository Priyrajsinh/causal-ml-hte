"""Minimalist theme for the Gradio CATE explorer.

Earlier iterations used a dark indigo/purple glass-morphism background
(rule C44 spirit) but that put dark text on a dark background and made
the readout unreadable. We now lean on Gradio's ``Soft`` theme tuned with
the indigo/purple palette - light background, dark text, indigo accent
on primary buttons. The visual brand is carried via the palette + the
primary button, not by a custom CSS layer.

The same theme is consumed by ``src/api/gradio_demo.py`` and by the
self-contained ``hf_space/app.py`` (rule C12).
"""

import gradio as gr

PRIMARY = "#6366f1"
SECONDARY = "#a855f7"


def get_theme() -> gr.themes.Soft:
    """Soft theme tuned to the indigo/purple palette."""
    return gr.themes.Soft(
        primary_hue=gr.themes.colors.indigo,
        secondary_hue=gr.themes.colors.purple,
        neutral_hue=gr.themes.colors.slate,
    )
