"""
Model Charts — structural layout only.
All colors and fonts are applied by the React frontend.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from ukraine_alerts.charts.theme import AXIS_STYLE, LAYOUT_BASE

# Regime band fills — semi-transparent, readable on dark backgrounds
# React's colorway handles trace colors; these fills use inline rgba for vrect
STATE_LABELS = ["Low Threat", "Elevated Threat", "Crisis/High Threat"]
STATE_FILL_COLORS = [
    "rgba(69, 123, 157, 0.15)",
    "rgba(244, 162, 97, 0.15)",
    "rgba(255, 51, 51, 0.15)",
]
STATE_MARKER_COLORS = ["#457B9D", "#F4A261", "#FF3333"]


def plot_prophet_forecast(forecast: pd.DataFrame, actual: pd.DataFrame, region: str = "") -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=pd.concat([forecast["ds"], forecast["ds"].iloc[::-1]]),
        y=pd.concat([forecast["yhat_upper"], forecast["yhat_lower"].iloc[::-1]]),
        fill="toself", fillcolor="rgba(136, 136, 136, 0.12)", line=dict(color="rgba(0,0,0,0)"),
        name="80% Confidence", hoverinfo="skip",
    ))

    fig.add_trace(go.Scatter(
        x=forecast["ds"], y=forecast["yhat"].clip(lower=0), name="Prophet Forecast", mode="lines",
        line=dict(width=2, dash="dot"),
        hovertemplate="<b>%{x}</b><br>Forecast: %{y:.1f}<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=actual["ds"], y=actual["alert_count"], name="Actual", mode="lines",
        line=dict(width=1.5),
        hovertemplate="<b>%{x}</b><br>Actual: %{y}<extra></extra>",
    ))

    cutoff = actual["ds"].max()
    fig.add_vline(
        x=str(cutoff)[:10], line=dict(color="rgba(237,237,237,0.3)", width=1.5, dash="dash"),
        annotation_text="Forecast start",
    )

    title = f"14-Day Alert Count Forecast — {region}" if region else "14-Day Alert Count Forecast"
    fig.update_layout(
        **LAYOUT_BASE,
        title=title,
        xaxis=dict(**AXIS_STYLE, title=""),
        yaxis=dict(**AXIS_STYLE, title="Daily Alert Count"),
    )
    return fig


def _add_regime_bands(fig: go.Figure, decoded: pd.DataFrame) -> None:
    if decoded.empty:
        return
    state_col = decoded["hmm_state"].values
    dates = decoded["ds"].values
    start_idx = 0
    current_state = state_col[0]
    for i in range(1, len(state_col)):
        if state_col[i] != current_state or i == len(state_col) - 1:
            end_idx = i if state_col[i] != current_state else i + 1
            fig.add_vrect(
                x0=str(dates[start_idx])[:10], x1=str(dates[min(end_idx, len(dates) - 1)])[:10],
                fillcolor=STATE_FILL_COLORS[current_state], line_width=0, layer="below",
            )
            start_idx = i
            current_state = state_col[i]


def plot_regime_overlay(decoded: pd.DataFrame, region: str = "") -> go.Figure:
    fig = go.Figure()
    _add_regime_bands(fig, decoded)

    fig.add_trace(go.Scatter(
        x=decoded["ds"], y=decoded["alert_count"], name="Daily Alerts", mode="lines",
        line=dict(width=1.5),
        hovertemplate="<b>%{x}</b><br>Alerts: %{y}<extra></extra>",
    ))

    for i, label in enumerate(STATE_LABELS):
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=12, color=STATE_MARKER_COLORS[i], symbol="square"),
            name=label, showlegend=True,
        ))

    title = f"Threat Regime Detection (HMM) — {region}" if region else "Threat Regime Detection (HMM)"
    fig.update_layout(
        **LAYOUT_BASE,
        title=title,
        xaxis=dict(**AXIS_STYLE, title=""),
        yaxis=dict(**AXIS_STYLE, title="Daily Alert Count"),
    )
    return fig
