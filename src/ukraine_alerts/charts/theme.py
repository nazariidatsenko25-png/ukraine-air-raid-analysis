"""
Dieter Rams (Functionalist) Theme for Plotly charts.

Less but better. Every element earns its place. Nothing decorative without function.
- Typography: Clean sans-serif.
- Color: Restrained. Dark monochromatic with a single functional accent (#FF3333).
- Details: No backgrounds, no grid lines unless strictly necessary. Flatness is the point.
"""

from __future__ import annotations

# Functionalist Color Palette
COLORS = {
    "background": "rgba(0,0,0,0)",  # Transparent, lets Streamlit #0A0A0A show through
    "text_primary": "#EDEDED",      # Off-white for legibility
    "text_muted": "#888888",        # Subdued for non-critical info
    "accent": "#FF3333",            # Single functional accent (Alert/Danger)
    "accent_muted": "#FF333333",    # 20% opacity for fills
    "surface": "#141414",           # Elevated surfaces
    "grid": "#222222",              # Extremely subtle grid
}

# Threat profile colors designed to contrast well on #0A0A0A while remaining flat
THREAT_COLORS = {
    "Strategic/Ballistic (MiG)": COLORS["accent"],
    "Loitering Munition (Shahed)": "#F4A261",
    "Tactical/Artillery": "#457B9D",
    "Unknown": COLORS["text_muted"],
}

# Regime colors
REGIME_COLORS = {
    "Low Threat": "#457B9D",
    "Elevated Threat": "#F4A261",
    "High Threat/Crisis": COLORS["accent"],
}

# Axis style: minimal, no bounding boxes, subtle grid
AXIS_STYLE = dict(
    showgrid=False,
    gridcolor=COLORS["grid"],
    gridwidth=1,
    zeroline=False,
    showline=False,
    color=COLORS["text_muted"],
    tickfont=dict(size=11, color=COLORS["text_muted"]),
    title_font=dict(size=12, color=COLORS["text_muted"]),
)

# Base Layout Template
LAYOUT_BASE = dict(
    template="plotly_dark",
    paper_bgcolor=COLORS["background"],
    plot_bgcolor=COLORS["background"],
    font=dict(family="sans-serif", color=COLORS["text_primary"], size=13),
    margin=dict(t=40, b=40, l=40, r=40),
    hovermode="closest",
    hoverlabel=dict(
        bgcolor=COLORS["surface"],
        font_size=12,
        font_family="sans-serif",
        bordercolor=COLORS["grid"],
    ),
)
