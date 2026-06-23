"""
Threat Charts module using Dieter Rams aesthetic.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from ukraine_alerts.charts.theme import AXIS_STYLE, LAYOUT_BASE, THREAT_COLORS


def plot_threat_scatter(waves: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        waves,
        x="duration_min", y="region_count", color="threat_profile",
        color_discrete_map=THREAT_COLORS, opacity=0.3,
        hover_data=["started_at"],
        title="Attack Waves by Duration and Geographic Spread",
        marginal_x="histogram", marginal_y="histogram",
    )

    fig.update_layout(
        **LAYOUT_BASE,
        xaxis=dict(**AXIS_STYLE, title="Wave Duration (minutes)", range=[0, 800]),
        yaxis=dict(**AXIS_STYLE, title="Number of Regions Affected"),
        legend_title_text="Inferred Threat Profile",
    )
    fig.update_traces(marker=dict(size=6, line=dict(width=0)))
    return fig

def plot_threat_timeline(waves: pd.DataFrame) -> go.Figure:
    waves_copy = waves.copy()
    waves_copy["month"] = waves_copy["started_at"].dt.to_period("M").dt.to_timestamp()
    monthly_threats = waves_copy.groupby(["month", "threat_profile"]).size().reset_index(name="count")

    fig = px.bar(
        monthly_threats,
        x="month", y="count", color="threat_profile",
        color_discrete_map=THREAT_COLORS,
        title="Monthly Attack Waves by Inferred Threat",
    )

    fig.update_layout(
        **LAYOUT_BASE,
        xaxis=dict(**AXIS_STYLE, title=""),
        yaxis=dict(**AXIS_STYLE, title="Number of Attack Waves"),
        legend_title_text="",
        barmode="stack",
    )
    return fig
