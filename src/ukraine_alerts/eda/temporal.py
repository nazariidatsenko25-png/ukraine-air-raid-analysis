"""
Temporal EDA module.

Produces time-based visualizations:
    - Daily alert count time series with 14-day rolling average
    - Hour-of-day × day-of-week heatmap (attack timing patterns)
    - Monthly alert count bar chart
    - Year-over-year comparison

All functions accept a clean DataFrame (output of preprocessing.build_clean_dataset)
and return Plotly figures ready for Streamlit embedding.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from ukraine_alerts.utils.constants import (
    COL_DATE,
    COL_DOW,
    COL_DURATION_MIN,
    COL_HOUR,
    COL_REGION,
    COL_START,
)

# ---------------------------------------------------------------------------
# Design tokens (consistent across all EDA modules)
# ---------------------------------------------------------------------------

COLORS = {
    "primary": "#E63946",       # Alert red
    "secondary": "#1D3557",     # Deep navy
    "accent": "#457B9D",        # Steel blue
    "muted": "#A8DADC",         # Pale teal
    "background": "#0D1117",    # Near-black background
    "surface": "#161B22",       # Card surface
    "border": "#30363D",        # Subtle border
    "text": "#E6EDF3",          # Primary text
    "text_muted": "#8B949E",    # Secondary text
}

LAYOUT_BASE = dict(
    paper_bgcolor=COLORS["background"],
    plot_bgcolor=COLORS["surface"],
    font=dict(family="'IBM Plex Mono', 'Courier New', monospace", color=COLORS["text"], size=12),
    title_font=dict(size=16, color=COLORS["text"]),
    margin=dict(l=60, r=30, t=60, b=60),
    hoverlabel=dict(
        bgcolor=COLORS["surface"],
        bordercolor=COLORS["border"],
        font=dict(color=COLORS["text"], family="'IBM Plex Mono', monospace"),
    ),
)

# Reusable axis style — merge into xaxis= / yaxis= kwarg as needed
AXIS_STYLE = dict(
    gridcolor=COLORS["border"],
    linecolor=COLORS["border"],
    tickcolor=COLORS["text_muted"],
    tickfont=dict(color=COLORS["text_muted"]),
    showgrid=True,
)

DOW_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def plot_daily_alert_counts(
    df: pd.DataFrame,
    region: str | None = None,
    rolling_window: int = 14,
) -> go.Figure:
    """
    Daily alert count time series with rolling average overlay.

    Args:
        df: Clean DataFrame from preprocessing.
        region: If specified, filter to one region. None = national aggregate.
        rolling_window: Days for rolling mean smoothing.

    Returns:
        Plotly figure with two traces: raw daily counts + rolling mean.
    """
    df_filtered = df[df[COL_REGION] == region].copy() if region else df.copy()
    title_suffix = region if region else "All Regions (National)"

    daily = (
        df_filtered.groupby(COL_DATE)
        .size()
        .reset_index(name="alert_count")
        .sort_values(COL_DATE)
    )
    daily["rolling_mean"] = daily["alert_count"].rolling(rolling_window, min_periods=1).mean()

    fig = go.Figure()

    # Raw counts as subtle bars
    fig.add_trace(go.Bar(
        x=daily[COL_DATE],
        y=daily["alert_count"],
        name="Daily alerts",
        marker_color=COLORS["accent"],
        marker_opacity=0.4,
        hovertemplate="<b>%{x}</b><br>Alerts: %{y}<extra></extra>",
    ))

    # Rolling mean as bold line
    fig.add_trace(go.Scatter(
        x=daily[COL_DATE],
        y=daily["rolling_mean"],
        name=f"{rolling_window}-day avg",
        line=dict(color=COLORS["primary"], width=2.5),
        hovertemplate="<b>%{x}</b><br>Rolling avg: %{y:.1f}<extra></extra>",
    ))

    fig.update_layout(
        **LAYOUT_BASE,
        title=f"Daily Alert Count — {title_suffix}",
        xaxis=dict(**AXIS_STYLE, title="Date"),
        yaxis=dict(**AXIS_STYLE, title="Number of Alerts"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color=COLORS["text_muted"]),
        ),
        barmode="overlay",
    )
    return fig


def plot_hourly_heatmap(
    df: pd.DataFrame,
    region: str | None = None,
) -> go.Figure:
    """
    Heatmap: hour-of-day (Kyiv local) × day-of-week.
    Reveals tactical timing patterns (pre-dawn raids, morning rush attacks).

    Args:
        df: Clean DataFrame from preprocessing.
        region: Filter to one region or aggregate nationally.

    Returns:
        Plotly heatmap figure.
    """
    df_filtered = df[df[COL_REGION] == region].copy() if region else df.copy()
    title_suffix = region if region else "All Regions"

    pivot = (
        df_filtered.groupby([COL_DOW, COL_HOUR])
        .size()
        .unstack(fill_value=0)
    )
    # Reorder days
    pivot = pivot.reindex([d for d in DOW_ORDER if d in pivot.index])

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[f"{h:02d}:00" for h in pivot.columns],
        y=pivot.index.tolist(),
        colorscale=[
            [0.0, COLORS["surface"]],
            [0.3, "#2A3F5A"],
            [0.6, COLORS["accent"]],
            [0.85, "#D62728"],
            [1.0, COLORS["primary"]],
        ],
        hoverongaps=False,
        hovertemplate="<b>%{y} %{x}</b><br>Alerts: %{z}<extra></extra>",
        colorbar=dict(
            title=dict(text="Alerts", font=dict(color=COLORS["text_muted"])),
            tickfont=dict(color=COLORS["text_muted"]),
            bgcolor=COLORS["surface"],
            bordercolor=COLORS["border"],
        ),
    ))

    layout = {**LAYOUT_BASE}
    layout["xaxis"] = {**AXIS_STYLE, "tickangle": -45, "side": "bottom", "title": "Hour of Day"}
    layout["yaxis"] = {**AXIS_STYLE}
    fig.update_layout(
        **layout,
        title=f"Alert Timing Heatmap (Kyiv Local Time) — {title_suffix}",
        yaxis_title="",
    )
    return fig


def plot_monthly_alert_counts(
    df: pd.DataFrame,
    region: str | None = None,
) -> go.Figure:
    """
    Monthly alert count bar chart showing escalation/de-escalation phases.

    Args:
        df: Clean DataFrame from preprocessing.
        region: Filter to one region or aggregate nationally.

    Returns:
        Plotly bar figure.
    """
    df_filtered = df[df[COL_REGION] == region].copy() if region else df.copy()
    title_suffix = region if region else "National"

    df_filtered["year_month"] = df_filtered[COL_START].dt.to_period("M").astype(str)
    monthly = (
        df_filtered.groupby("year_month")
        .size()
        .reset_index(name="alert_count")
        .sort_values("year_month")
    )

    fig = go.Figure(go.Bar(
        x=monthly["year_month"],
        y=monthly["alert_count"],
        marker=dict(
            color=monthly["alert_count"],
            colorscale=[
                [0, COLORS["accent"]],
                [0.5, "#D45F5F"],
                [1, COLORS["primary"]],
            ],
            showscale=False,
        ),
        hovertemplate="<b>%{x}</b><br>Alerts: %{y}<extra></extra>",
    ))

    fig.update_layout(
        **LAYOUT_BASE,
        title=f"Monthly Alert Volume — {title_suffix}",
        xaxis=dict(**AXIS_STYLE, tickangle=-45, title="Month"),
        yaxis=dict(**AXIS_STYLE, title="Number of Alerts"),
    )
    return fig


def plot_duration_distribution(
    df: pd.DataFrame,
    region: str | None = None,
    max_duration_minutes: float = 300.0,
) -> go.Figure:
    """
    Alert duration distribution (histogram + KDE-style smoothing).
    Clips extreme outliers (Luhansk/Crimea permanent sirens) for readability.

    Args:
        df: Clean DataFrame from preprocessing.
        region: Filter to one region or aggregate nationally.
        max_duration_minutes: Clip outliers above this value.

    Returns:
        Plotly histogram figure.
    """
    df_filtered = df[df[COL_REGION] == region].copy() if region else df.copy()
    title_suffix = region if region else "All Regions"

    durations = df_filtered[COL_DURATION_MIN].clip(upper=max_duration_minutes)
    n_clipped = (df_filtered[COL_DURATION_MIN] > max_duration_minutes).sum()

    fig = go.Figure(go.Histogram(
        x=durations,
        nbinsx=80,
        marker=dict(
            color=COLORS["accent"],
            opacity=0.8,
            line=dict(color=COLORS["border"], width=0.5),
        ),
        hovertemplate="Duration: %{x:.0f} min<br>Count: %{y}<extra></extra>",
    ))

    annotation_text = (
        f"Clipped {n_clipped:,} extreme outliers (>{max_duration_minutes:.0f} min)<br>"
        f"Median: {df_filtered[COL_DURATION_MIN].median():.1f} min"
    )

    fig.update_layout(
        **LAYOUT_BASE,
        title=f"Alert Duration Distribution — {title_suffix}",
        xaxis=dict(**AXIS_STYLE, title="Duration (minutes)"),
        yaxis=dict(**AXIS_STYLE, title="Count"),
        annotations=[dict(
            x=0.98, y=0.95, xref="paper", yref="paper",
            text=annotation_text,
            showarrow=False,
            font=dict(color=COLORS["text_muted"], size=11),
            align="right",
            bgcolor=COLORS["surface"],
            bordercolor=COLORS["border"],
            borderwidth=1,
        )],
    )
    return fig
