"""
Cascade Charts module using Dieter Rams aesthetic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ukraine_alerts.charts.theme import AXIS_STYLE, COLORS, LAYOUT_BASE
from ukraine_alerts.utils.constants import CASCADE_WINDOW_HOURS


def plot_cascade_heatmap(matrix: pd.DataFrame, title: str = "Cross-Region Alert Cascade Probability") -> go.Figure:
    short_names = [r.replace(" oblast", "").replace(" region", "") for r in matrix.index]
    m = matrix.values.copy()
    np.fill_diagonal(m, np.nan)

    fig = go.Figure(go.Heatmap(
        z=m, x=short_names, y=short_names,
        colorscale=[[0.0, COLORS["background"]], [1.0, COLORS["accent"]]],
        zmid=0.3, hoverongaps=False,
        hovertemplate="Trigger: <b>%{y}</b><br>Response: <b>%{x}</b><br>P(response within window): <b>%{z:.1%}</b><extra></extra>",
        colorbar=dict(title=dict(text="P(cascade)", font=dict(color=COLORS["text_muted"])), tickformat=".0%", tickfont=dict(color=COLORS["text_muted"])),
    ))

    fig.update_layout(
        **LAYOUT_BASE,
        title=title,
        xaxis=dict(**AXIS_STYLE, tickangle=-45, side="bottom"),
        yaxis=dict(**AXIS_STYLE, autorange="reversed"),
        height=700,
    )
    return fig

def plot_top_cascade_pairs(matrix: pd.DataFrame, top_n: int = 15, window_hours: float = CASCADE_WINDOW_HOURS) -> go.Figure:
    pairs = []
    for trigger in matrix.index:
        for response in matrix.columns:
            if trigger == response:
                continue
            prob = matrix.loc[trigger, response]
            if not np.isnan(prob) and prob > 0:
                t_short = trigger.replace(" oblast", "")
                r_short = response.replace(" oblast", "")
                pairs.append({"pair": f"{t_short} → {r_short}", "probability": prob})

    pairs_df = pd.DataFrame(pairs).sort_values("probability", ascending=True).tail(top_n)

    fig = go.Figure(go.Bar(
        x=pairs_df["probability"], y=pairs_df["pair"], orientation="h",
        marker_color=COLORS["accent"],
        hovertemplate="<b>%{y}</b><br>P(cascade): %{x:.1%}<extra></extra>",
    ))

    fig.update_layout(
        **LAYOUT_BASE,
        title=f"Top {top_n} Cascade Pairs (within {window_hours:.0f}h window)",
        xaxis=dict(**AXIS_STYLE, tickformat=".0%", title="Conditional Probability"),
        yaxis=dict(**AXIS_STYLE),
        height=max(400, top_n * 30),
    )
    return fig

def plot_secondary_strike_curve(curve_df: pd.DataFrame, trigger_region: str) -> go.Figure:
    region_short = trigger_region.replace(" oblast", "")

    fig = go.Figure(go.Scatter(
        x=curve_df["window_hours"], y=curve_df["probability"], mode="lines+markers",
        line=dict(color=COLORS["accent"], width=2.5), marker=dict(color=COLORS["accent"], size=8),
        fill="tozeroy", fillcolor=COLORS["accent_muted"],
        hovertemplate="Within <b>%{x}h</b>: P(secondary strike) = <b>%{y:.1%}</b><extra></extra>",
    ))

    fig.update_layout(
        **LAYOUT_BASE,
        title=f"National Secondary Strike Probability After Alert in {region_short}",
        xaxis=dict(**AXIS_STYLE, title="Time Window (hours)"),
        yaxis=dict(**AXIS_STYLE, tickformat=".0%", title="P(at least one secondary alert nationally)"),
    )
    return fig
