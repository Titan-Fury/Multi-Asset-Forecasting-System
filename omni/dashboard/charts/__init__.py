"""Cutting-edge chart helpers — TradingView, D3, Three.js embeds.

Each module returns a self-contained HTML+JS string and helper renderers
that mount cleanly inside a Streamlit page.

Modules:
  lwchart   — TradingView Lightweight-Charts (price + volume candles)
  cosmos3d  — Three.js 3D signal cosmos
  corrgraph — D3 force-directed correlation network

Render helper
-------------
`render_html(html, *, height)` is the one place we choose the Streamlit
embed API. We prefer ``st.html(..., unsafe_allow_javascript=True)`` which
lands the markup directly in the page DOM and is the supported path going
forward (``st.components.v1.html`` is deprecated for removal after
2026-06-01). On Streamlit < 1.40 the function falls back to the legacy
``components.v1.html``. The ``height`` kwarg is preserved for the legacy
path; with ``st.html`` the inner HTML/CSS already sets canvas dimensions
so the argument is ignored.
"""
from __future__ import annotations


def render_html(html: str, *, height: int = 480) -> None:
    """Mount a self-contained HTML+JS string into the current Streamlit page.

    Always routes through ``streamlit.components.v1.html`` (iframe) when the
    payload contains ``<script>`` tags — even on Streamlit >= 1.40 where
    ``st.html(..., unsafe_allow_javascript=True)`` exists. Reason: scripts
    inserted into the parent DOM via ``innerHTML`` are NOT executed by the
    browser (HTML spec), so the candle/3D/correlation charts come up as
    empty containers if we go through ``st.html``. The iframe path always
    works because the iframe document is parsed fresh.
    """
    import streamlit as st

    has_script = "<script" in html.lower()

    if not has_script and hasattr(st, "html"):
        # Pure HTML/CSS — st.html is fine and renders inline (no iframe).
        try:
            st.html(html)
            return
        except Exception:
            pass

    import streamlit.components.v1 as components
    components.html(html, height=height + 20, scrolling=False)
