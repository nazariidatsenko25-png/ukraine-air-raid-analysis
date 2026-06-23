"""
EDA Charts module using Dieter Rams aesthetic.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from ukraine_alerts.charts.theme import AXIS_STYLE, COLORS, LAYOUT_BASE
from ukraine_alerts.eda.regional import PERMANENT_SIREN_REGIONS, build_region_summary
from ukraine_alerts.utils.constants import (
    COL_DATE,
    COL_DOW,
    COL_DURATION_MIN,
    COL_HOUR,
    COL_REGION,
    COL_START,
)

DOW_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

def plot_daily_alert_counts(df: pd.DataFrame, region: str | None = None, rolling_window: int = 14) -> go.Figure:
    df_filtered = df[df[COL_REGION] == region].copy() if region else df.copy()
    title_suffix = region if region else "All Regions (National)"

    daily = df_filtered.groupby(COL_DATE).size().reset_index(name="alert_count").sort_values(COL_DATE)
    daily["rolling_mean"] = daily["alert_count"].rolling(rolling_window, min_periods=1).mean()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=daily[COL_DATE], y=daily["alert_count"], name="Daily alerts",
        marker_color=COLORS["accent"], marker_opacity=0.4,
        hovertemplate="<b>%{x}</b><br>Alerts: %{y}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=daily[COL_DATE], y=daily["rolling_mean"], name=f"{rolling_window}-day avg",
        line=dict(color=COLORS["accent"], width=2.5),
        hovertemplate="<b>%{x}</b><br>Rolling avg: %{y:.1f}<extra></extra>",
    ))

    fig.update_layout(
        **LAYOUT_BASE,
        title=f"Daily Alert Count — {title_suffix}",
        xaxis=dict(**AXIS_STYLE, title="Date"),
        yaxis=dict(**AXIS_STYLE, title="Number of Alerts"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color=COLORS["text_muted"])),
        barmode="overlay",
    )
    return fig

def plot_hourly_heatmap(df: pd.DataFrame, region: str | None = None) -> go.Figure:
    df_filtered = df[df[COL_REGION] == region].copy() if region else df.copy()
    title_suffix = region if region else "All Regions"

    pivot = df_filtered.groupby([COL_DOW, COL_HOUR]).size().unstack(fill_value=0)
    pivot = pivot.reindex([d for d in DOW_ORDER if d in pivot.index])

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[f"{h:02d}:00" for h in pivot.columns],
        y=pivot.index.tolist(),
        colorscale=[
            [0.0, COLORS["background"]],
            [1.0, COLORS["accent"]],
        ],
        hoverongaps=False,
        hovertemplate="<b>%{y} %{x}</b><br>Alerts: %{z}<extra></extra>",
        colorbar=dict(title=dict(text="Alerts", font=dict(color=COLORS["text_muted"])), tickfont=dict(color=COLORS["text_muted"])),
    ))

    layout = {**LAYOUT_BASE}
    layout["xaxis"] = {**AXIS_STYLE, "tickangle": -45, "side": "bottom", "title": "Hour of Day"}
    layout["yaxis"] = {**AXIS_STYLE}
    fig.update_layout(**layout, title=f"Alert Timing Heatmap (Kyiv Local Time) — {title_suffix}", yaxis_title="")
    return fig

def plot_monthly_alert_counts(df: pd.DataFrame, region: str | None = None) -> go.Figure:
    df_filtered = df[df[COL_REGION] == region].copy() if region else df.copy()
    title_suffix = region if region else "National"

    df_filtered["year_month"] = df_filtered[COL_START].dt.to_period("M").astype(str)
    monthly = df_filtered.groupby("year_month").size().reset_index(name="alert_count").sort_values("year_month")

    fig = go.Figure(go.Bar(
        x=monthly["year_month"], y=monthly["alert_count"],
        marker_color=COLORS["accent"],
        hovertemplate="<b>%{x}</b><br>Alerts: %{y}<extra></extra>",
    ))

    fig.update_layout(
        **LAYOUT_BASE,
        title=f"Monthly Alert Volume — {title_suffix}",
        xaxis=dict(**AXIS_STYLE, tickangle=-45, title="Month"),
        yaxis=dict(**AXIS_STYLE, title="Number of Alerts"),
    )
    return fig

def plot_duration_distribution(df: pd.DataFrame, region: str | None = None, max_duration_minutes: float = 300.0) -> go.Figure:
    df_filtered = df[df[COL_REGION] == region].copy() if region else df.copy()
    title_suffix = region if region else "All Regions"

    durations = df_filtered[COL_DURATION_MIN].clip(upper=max_duration_minutes)
    n_clipped = (df_filtered[COL_DURATION_MIN] > max_duration_minutes).sum()

    fig = go.Figure(go.Histogram(
        x=durations, nbinsx=80,
        marker=dict(color=COLORS["accent"], opacity=0.8, line=dict(color=COLORS["background"], width=0.5)),
        hovertemplate="Duration: %{x:.0f} min<br>Count: %{y}<extra></extra>",
    ))

    annotation_text = f"Clipped {n_clipped:,} extreme outliers (>{max_duration_minutes:.0f} min)<br>Median: {df_filtered[COL_DURATION_MIN].median():.1f} min"
    fig.update_layout(
        **LAYOUT_BASE,
        title=f"Alert Duration Distribution — {title_suffix}",
        xaxis=dict(**AXIS_STYLE, title="Duration (minutes)"),
        yaxis=dict(**AXIS_STYLE, title="Count"),
        annotations=[dict(x=0.98, y=0.95, xref="paper", yref="paper", text=annotation_text, showarrow=False, font=dict(color=COLORS["text_muted"], size=11), align="right")],
    )
    return fig

def plot_region_alert_ranking(df: pd.DataFrame, top_n: int = 25, exclude_permanent: bool = False) -> go.Figure:
    summary = build_region_summary(df)
    if exclude_permanent:
        summary = summary[~summary[COL_REGION].isin(PERMANENT_SIREN_REGIONS)]
    summary = summary.head(top_n)

    fig = go.Figure(go.Bar(
        x=summary["alert_count"], y=summary[COL_REGION], orientation="h",
        marker_color=COLORS["accent"],
        hovertemplate="<b>%{y}</b><br>Total alerts: %{x:,}<br><extra></extra>",
    ))

    fig.update_layout(
        **LAYOUT_BASE,
        title="Alert Count by Region",
        xaxis=dict(**AXIS_STYLE, title="Total Alerts"),
        yaxis=dict(**AXIS_STYLE, autorange="reversed", categoryorder="total ascending"),
        height=max(400, top_n * 28),
    )
    return fig

def plot_region_duration_comparison(df: pd.DataFrame, exclude_permanent: bool = True) -> go.Figure:
    df_filtered = df.copy()
    if exclude_permanent:
        df_filtered = df_filtered[~df_filtered[COL_REGION].isin(PERMANENT_SIREN_REGIONS)]

    summary = build_region_summary(df_filtered).sort_values("avg_duration_min", ascending=False)

    fig = go.Figure(go.Bar(
        x=summary["avg_duration_min"], y=summary[COL_REGION], orientation="h",
        marker_color=COLORS["accent"],
        hovertemplate="<b>%{y}</b><br>Median duration: %{x:.1f} min<br><extra></extra>",
    ))

    fig.update_layout(
        **LAYOUT_BASE,
        title="Median Alert Duration by Region (excl. permanent sirens)",
        xaxis=dict(**AXIS_STYLE, title="Median Duration (minutes)"),
        yaxis=dict(**AXIS_STYLE, autorange="reversed"),
        height=max(400, len(summary) * 28),
    )
    return fig

def plot_regional_intensity_over_time(df: pd.DataFrame, top_n_regions: int = 6, freq: str = "W") -> go.Figure:
    top_regions = df.groupby(COL_REGION).size().nlargest(top_n_regions).index.tolist()
    df_top = df[df[COL_REGION].isin(top_regions)].copy()
    df_top["period"] = df_top[COL_START].dt.to_period(freq).dt.to_timestamp()
    weekly = df_top.groupby([COL_REGION, "period"]).size().reset_index(name="count").sort_values("period")

    palette = [COLORS["text_primary"], COLORS["accent"], "#555555", "#888888", "#BBBBBB", "#333333"]

    fig = go.Figure()
    for i, region in enumerate(top_regions):
        region_data = weekly[weekly[COL_REGION] == region]
        fig.add_trace(go.Scatter(
            x=region_data["period"], y=region_data["count"], name=region.replace(" oblast", ""), mode="lines",
            line=dict(color=palette[i % len(palette)], width=2),
            hovertemplate=f"<b>{region}</b><br>Period: %{{x}}<br>Alerts: %{{y}}<extra></extra>",
        ))

    freq_label = {"W": "Weekly", "ME": "Monthly"}.get(freq, freq)
    fig.update_layout(
        **LAYOUT_BASE,
        title=f"{freq_label} Alert Counts — Top {top_n_regions} Regions",
        xaxis=dict(**AXIS_STYLE), yaxis=dict(**AXIS_STYLE, title="Alert Count"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color=COLORS["text_muted"], size=10)),
    )
    return fig

def plot_region_treemap(df: pd.DataFrame) -> go.Figure:
    summary = build_region_summary(df)

    fig = go.Figure(go.Treemap(
        labels=summary[COL_REGION], parents=["Ukraine"] * len(summary), values=summary["alert_count"],
        texttemplate="<b>%{label}</b><br>%{value:,}", textfont=dict(color=COLORS["text_primary"], size=11),
        marker=dict(colors=summary["alert_count"], colorscale=[[0, COLORS["background"]], [1, COLORS["accent"]]], line=dict(color=COLORS["background"], width=2)),
        hovertemplate="<b>%{label}</b><br>Alerts: %{value:,}<extra></extra>",
    ))

    layout = {**LAYOUT_BASE, "margin": dict(l=0, r=0, t=40, b=0)}
    fig.update_layout(**layout, title="Alert Share by Region")
    return fig
