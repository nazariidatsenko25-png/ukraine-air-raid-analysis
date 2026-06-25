"""
Backend chart theme — structural layout only.

All aesthetic styling (colors, fonts, hover labels, margins, palettes) is owned
by the React frontend (Chart.tsx baseLayout). This module provides only structural
defaults that convey meaning (axis labels, hover templates, titles) — not appearance.

The frontend deep-merges its own aesthetic config on top of whatever Python sends,
so any bgcolor/font/color values set here would be overwritten by React anyway.
"""

from __future__ import annotations

# Structural axis defaults — no colors, just layout behavior
AXIS_STYLE = dict(
    showgrid=True,
    zeroline=False,
    showline=False,
    automargin=True,
)

# Structural layout base — no bgcolor, no fonts, no palette
LAYOUT_BASE = dict(
    autosize=True,
    hovermode="closest",
)

# Threat profile labels (used for legend text in traces)
THREAT_LABELS = {
    "Strategic/Ballistic (MiG)": 0,
    "Loitering Munition (Shahed)": 1,
    "Tactical/Artillery": 2,
    "Unknown": 3,
}

# Regime labels (used for trace names in HMM charts)
REGIME_LABELS = {
    "Low Threat": "Low Threat",
    "Elevated Threat": "Elevated Threat",
    "High Threat/Crisis": "High Threat/Crisis",
}
