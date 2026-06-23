"""
Regional EDA module.

Produces region-comparative visualizations:
    - Alert count ranking (horizontal bar chart)
    - Average duration per region
    - Alert intensity over time per region (filled area chart)
    - Region alert share (treemap)

All functions accept a clean DataFrame (output of preprocessing.build_clean_dataset).
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from ukraine_alerts.eda.temporal import AXIS_STYLE, COLORS, LAYOUT_BASE
from ukraine_alerts.utils.constants import (
    COL_DURATION_MIN,
    COL_IS_IMPUTED,
    COL_REGION,
    COL_START,
)

# Regions with known permanent/long-running sirens — optionally excluded from
# duration statistics to avoid distorting averages.
PERMANENT_SIREN_REGIONS = {"Luhanska oblast", "Crimea"}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def build_region_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Build a per-region summary DataFrame used by multiple plots."""
    summary = (
        df.groupby(COL_REGION)
        .agg(
            alert_count=(COL_REGION, "count"),
            avg_duration_min=(COL_DURATION_MIN, "median"),  # median more robust than mean
            total_duration_hours=(COL_DURATION_MIN, lambda x: x.sum() / 60),
            pct_imputed=(COL_IS_IMPUTED, "mean"),
        )
        .reset_index()
        .sort_values("alert_count", ascending=False)
    )
    return summary


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def plot_region_alert_ranking(
    df: pd.DataFrame,
    top_n: int = 25,
    exclude_permanent: bool = False,
) -> go.Figure:
    """
    Horizontal bar chart ranking regions by total alert count.

    Args:
        df: Clean DataFrame from preprocessing.
        top_n: Maximum number of regions to show.
        exclude_permanent: If True, exclude Luhansk/Crimea permanent sirens.

    Returns:
        Plotly horizontal bar figure.
    """
    summary = build_region_summary(df)
    if exclude_permanent:
        summary = summary[~summary[COL_REGION].isin(PERMANENT_SIREN_REGIONS)]
    summary = summary.head(top_n)

    # Color by alert count
    max_count = summary["alert_count"].max()
    colors = [
        f"rgba(230, 57, 70, {0.4 + 0.6 * (c / max_count):.2f})"
        for c in summary["alert_count"]
    ]

    fig = go.Figure(go.Bar(
        x=summary["alert_count"],
        y=summary[COL_REGION],
        orientation="h",
        marker_color=colors,
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Total alerts: %{x:,}<br>"
            "<extra></extra>"
        ),
    ))

    fig.update_layout(
        **LAYOUT_BASE,
        title="Alert Count by Region",
        xaxis=dict(**AXIS_STYLE, title="Total Alerts"),
        yaxis=dict(
            **AXIS_STYLE,
            autorange="reversed",
            categoryorder="total ascending",
        ),
        height=max(400, top_n * 28),
    )
    return fig


def plot_region_duration_comparison(
    df: pd.DataFrame,
    exclude_permanent: bool = True,
) -> go.Figure:
    """
    Horizontal bar chart comparing median alert duration per region.
    Excludes permanent siren regions by default to avoid misleading statistics.

    Args:
        df: Clean DataFrame from preprocessing.
        exclude_permanent: If True, exclude Luhansk/Crimea.

    Returns:
        Plotly horizontal bar figure.
    """
    df_filtered = df.copy()
    if exclude_permanent:
        df_filtered = df_filtered[~df_filtered[COL_REGION].isin(PERMANENT_SIREN_REGIONS)]

    summary = build_region_summary(df_filtered).sort_values("avg_duration_min", ascending=False)

    fig = go.Figure(go.Bar(
        x=summary["avg_duration_min"],
        y=summary[COL_REGION],
        orientation="h",
        marker=dict(
            color=summary["avg_duration_min"],
            colorscale=[
                [0, COLORS["accent"]],
                [0.5, "#D45F5F"],
                [1, COLORS["primary"]],
            ],
            showscale=True,
            colorbar=dict(
                title=dict(text="Minutes", font=dict(color=COLORS["text_muted"])),
                tickfont=dict(color=COLORS["text_muted"]),
                bgcolor=COLORS["surface"],
                bordercolor=COLORS["border"],
            ),
        ),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Median duration: %{x:.1f} min<br>"
            "<extra></extra>"
        ),
    ))

    fig.update_layout(
        **LAYOUT_BASE,
        title="Median Alert Duration by Region (excl. permanent sirens)",
        xaxis=dict(**AXIS_STYLE, title="Median Duration (minutes)"),
        yaxis=dict(**AXIS_STYLE, autorange="reversed"),
        height=max(400, len(summary) * 28),
    )
    return fig


def plot_regional_intensity_over_time(
    df: pd.DataFrame,
    top_n_regions: int = 6,
    freq: str = "W",
) -> go.Figure:
    """
    Multi-line chart: weekly alert counts for the top N most-affected regions.
    Reveals which regions escalate/de-escalate relative to each other.

    Args:
        df: Clean DataFrame from preprocessing.
        top_n_regions: Number of top regions by total count to show.
        freq: Resampling frequency ('W' = weekly, 'ME' = monthly).

    Returns:
        Plotly line figure.
    """
    top_regions = (
        df.groupby(COL_REGION)
        .size()
        .nlargest(top_n_regions)
        .index.tolist()
    )
    df_top = df[df[COL_REGION].isin(top_regions)].copy()

    # Resample by region
    df_top["period"] = df_top[COL_START].dt.to_period(freq).dt.to_timestamp()
    weekly = (
        df_top.groupby([COL_REGION, "period"])
        .size()
        .reset_index(name="count")
        .sort_values("period")
    )

    palette = [
        COLORS["primary"], COLORS["accent"], "#F4A261",
        "#E9C46A", "#2A9D8F", "#9B5DE5",
    ]

    fig = go.Figure()
    for i, region in enumerate(top_regions):
        region_data = weekly[weekly[COL_REGION] == region]
        fig.add_trace(go.Scatter(
            x=region_data["period"],
            y=region_data["count"],
            name=region.replace(" oblast", ""),
            mode="lines",
            line=dict(color=palette[i % len(palette)], width=2),
            hovertemplate=f"<b>{region}</b><br>Period: %{{x}}<br>Alerts: %{{y}}<extra></extra>",
        ))

    freq_label = {"W": "Weekly", "ME": "Monthly"}.get(freq, freq)
    fig.update_layout(
        **LAYOUT_BASE,
        title=f"{freq_label} Alert Counts — Top {top_n_regions} Regions",
        xaxis=dict(**AXIS_STYLE),
        yaxis=dict(**AXIS_STYLE, title="Alert Count"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color=COLORS["text_muted"], size=10),
        ),
    )
    return fig


def plot_region_treemap(df: pd.DataFrame) -> go.Figure:
    """
    Treemap showing each region's share of total national alert volume.

    Args:
        df: Clean DataFrame from preprocessing.

    Returns:
        Plotly treemap figure.
    """
    summary = build_region_summary(df)

    fig = go.Figure(go.Treemap(
        labels=summary[COL_REGION],
        parents=["Ukraine"] * len(summary),
        values=summary["alert_count"],
        texttemplate="<b>%{label}</b><br>%{value:,}",
        textfont=dict(color=COLORS["text"], size=11),
        marker=dict(
            colorscale=[
                [0, COLORS["surface"]],
                [0.4, COLORS["accent"]],
                [1, COLORS["primary"]],
            ],
            colors=summary["alert_count"],
            showscale=False,
            line=dict(color=COLORS["border"], width=1),
        ),
        hovertemplate="<b>%{label}</b><br>Alerts: %{value:,}<extra></extra>",
    ))

    layout = {**LAYOUT_BASE, "margin": dict(l=10, r=10, t=60, b=10)}
    fig.update_layout(
        **layout,
        title="Alert Share by Region",
    )
    return fig
