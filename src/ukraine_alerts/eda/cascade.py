"""
Cascade analysis module.

Computes conditional probability of alert in Region B within N hours
given an alert just occurred in Region A.

This is the analytically distinctive feature of this project. Standard
time series tools don't capture cross-region excitation. This module
implements a simplified, interpretable version of the Hawkes process
excitation kernel using empirical conditional probabilities.

Usage:
    from ukraine_alerts.eda.cascade import (
        compute_cascade_matrix,
        plot_cascade_heatmap,
        plot_secondary_strike_probability,
    )

    matrix = compute_cascade_matrix(df_clean, window_hours=3)
    fig = plot_cascade_heatmap(matrix)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ukraine_alerts.eda.temporal import AXIS_STYLE, COLORS, LAYOUT_BASE
from ukraine_alerts.utils.constants import (
    CASCADE_WINDOW_HOURS,
    COL_REGION,
    COL_START,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def compute_cascade_matrix(
    df: pd.DataFrame,
    window_hours: float = CASCADE_WINDOW_HOURS,
    min_trigger_events: int = 10,
) -> pd.DataFrame:
    """
    Compute an N×N conditional probability matrix where entry [A, B] represents:

        P(alert in region B within `window_hours` | alert just started in region A)

    Algorithm:
        1. For each alert in region A (the "trigger"), find all alerts in other
           regions that start within [0, window_hours] after the trigger.
        2. Count how many triggers in A produced at least one response in B.
        3. Divide by total triggers in A to get the conditional probability.

    Complexity: O(n * k) where n = total alerts, k = avg alerts per window.
    For 100k alerts with a 4h window this runs in ~5-15 seconds.

    Args:
        df: Clean DataFrame sorted by started_at.
        window_hours: Look-ahead window in hours.
        min_trigger_events: Minimum number of triggers from a region to include
                            it in the matrix (avoids noisy estimates from rare regions).

    Returns:
        DataFrame with shape (n_regions × n_regions) of conditional probabilities.
        Index = trigger region, columns = response region.
    """
    logger.info(
        "Computing cascade matrix with %.1fh window on %d alerts",
        window_hours, len(df),
    )

    df_sorted = df[[COL_REGION, COL_START]].sort_values(COL_START).reset_index(drop=True)
    window_ns = int(window_hours * 3600 * 1e9)  # nanoseconds for pandas timedelta

    regions = df_sorted[COL_REGION].unique().tolist()

    # Count triggers per region for filtering
    trigger_counts = df_sorted[COL_REGION].value_counts()
    active_regions = [r for r in regions if trigger_counts.get(r, 0) >= min_trigger_events]
    active_regions = sorted(active_regions)

    # Initialize: [trigger_region][response_region] = set of trigger indices that produced a response
    n = len(active_regions)
    region_idx = {r: i for i, r in enumerate(active_regions)}

    # co_occurrence[i][j] = number of triggers from region i followed by alert in region j
    co_occurrence = np.zeros((n, n), dtype=np.int32)
    trigger_total = np.zeros(n, dtype=np.int32)

    starts = df_sorted[COL_START].values  # numpy datetime64 array
    region_vals = df_sorted[COL_REGION].values

    for idx in range(len(df_sorted)):
        trigger_region = region_vals[idx]
        if trigger_region not in region_idx:
            continue

        tri = region_idx[trigger_region]
        trigger_total[tri] += 1

        trigger_time = starts[idx]
        window_end = trigger_time + np.timedelta64(window_ns, "ns")

        # Binary search: find first alert after trigger
        lo = idx + 1
        # Find upper bound of window
        hi = np.searchsorted(starts, window_end, side="right")

        if lo >= hi:
            continue

        # Check all alerts within window
        window_regions = region_vals[lo:hi]
        for resp_region in set(window_regions):
            if resp_region == trigger_region:
                continue
            if resp_region not in region_idx:
                continue
            co_occurrence[tri, region_idx[resp_region]] += 1

    # Normalize: conditional probability
    with np.errstate(divide="ignore", invalid="ignore"):
        prob_matrix = np.where(
            trigger_total[:, None] > 0,
            co_occurrence / trigger_total[:, None],
            0.0,
        )

    result = pd.DataFrame(prob_matrix, index=active_regions, columns=active_regions)
    logger.info("Cascade matrix computed: %d×%d regions", n, n)
    return result


def compute_secondary_strike_curve(
    df: pd.DataFrame,
    trigger_region: str,
    windows_hours: list[float] | None = None,
) -> pd.DataFrame:
    """
    For a specific trigger region, compute P(secondary strike anywhere | trigger)
    as a function of time window length.

    Useful for answering: "Within how many hours after an alert in Kyiv is the
    national threat level elevated?"

    Args:
        df: Clean DataFrame.
        trigger_region: The region to use as trigger.
        windows_hours: List of window sizes in hours to evaluate.

    Returns:
        DataFrame with columns ['window_hours', 'probability'].
    """
    if windows_hours is None:
        windows_hours = [0.5, 1, 2, 3, 4, 6, 8, 12, 24]

    results = []
    for w in windows_hours:
        matrix = compute_cascade_matrix(df, window_hours=w, min_trigger_events=5)
        if trigger_region not in matrix.index:
            results.append({"window_hours": w, "probability": np.nan})
            continue
        # Probability that ANY other region fires within window
        row = matrix.loc[trigger_region].drop(trigger_region, errors="ignore")
        # P(at least one response) ≈ 1 - P(no response) = 1 - prod(1 - p_i)
        # (assuming independence — approximation, but interpretable)
        p_any = 1 - np.prod(1 - row.clip(0, 1).values)
        results.append({"window_hours": w, "probability": p_any})

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------


def plot_cascade_heatmap(
    matrix: pd.DataFrame,
    title: str = "Cross-Region Alert Cascade Probability",
) -> go.Figure:
    """
    Heatmap of the cascade matrix.
    Entry [row, col] = P(alert in column region within N hours | alert in row region).

    Args:
        matrix: Output of compute_cascade_matrix().
        title: Plot title.

    Returns:
        Plotly heatmap figure.
    """
    # Shorten region names for readability
    short_names = [r.replace(" oblast", "").replace(" region", "") for r in matrix.index]

    # Zero out the diagonal (self-excitation not meaningful here)
    m = matrix.values.copy()
    np.fill_diagonal(m, np.nan)

    fig = go.Figure(go.Heatmap(
        z=m,
        x=short_names,
        y=short_names,
        colorscale=[
            [0.0,  COLORS["surface"]],
            [0.25, "#1A3A5C"],
            [0.5,  COLORS["accent"]],
            [0.75, "#C0392B"],
            [1.0,  COLORS["primary"]],
        ],
        zmid=0.3,
        hoverongaps=False,
        hovertemplate=(
            "Trigger: <b>%{y}</b><br>"
            "Response: <b>%{x}</b><br>"
            "P(response within window): <b>%{z:.1%}</b>"
            "<extra></extra>"
        ),
        colorbar=dict(
            title=dict(text="P(cascade)", font=dict(color=COLORS["text_muted"])),
            tickformat=".0%",
            tickfont=dict(color=COLORS["text_muted"]),
            bgcolor=COLORS["surface"],
            bordercolor=COLORS["border"],
        ),
    ))

    fig.update_layout(
        **LAYOUT_BASE,
        title=title,
        xaxis=dict(
            **AXIS_STYLE,
            tickangle=-45,
            side="bottom",
        ),
        yaxis=dict(
            **AXIS_STYLE,
            autorange="reversed",
        ),
        height=700,
    )
    return fig


def plot_top_cascade_pairs(
    matrix: pd.DataFrame,
    top_n: int = 15,
    window_hours: float = CASCADE_WINDOW_HOURS,
) -> go.Figure:
    """
    Bar chart of the N highest-probability trigger→response pairs.
    More readable than the full heatmap for quick insight extraction.

    Args:
        matrix: Output of compute_cascade_matrix().
        top_n: Number of top pairs to show.
        window_hours: Window size (used in label only).

    Returns:
        Plotly bar figure.
    """
    # Flatten matrix, drop diagonal
    pairs = []
    for trigger in matrix.index:
        for response in matrix.columns:
            if trigger == response:
                continue
            prob = matrix.loc[trigger, response]
            if not np.isnan(prob) and prob > 0:
                t_short = trigger.replace(" oblast", "")
                r_short = response.replace(" oblast", "")
                pairs.append({
                    "pair": f"{t_short} → {r_short}",
                    "probability": prob,
                    "trigger": t_short,
                    "response": r_short,
                })

    pairs_df = (
        pd.DataFrame(pairs)
        .sort_values("probability", ascending=True)
        .tail(top_n)
    )

    fig = go.Figure(go.Bar(
        x=pairs_df["probability"],
        y=pairs_df["pair"],
        orientation="h",
        marker=dict(
            color=pairs_df["probability"],
            colorscale=[
                [0, COLORS["accent"]],
                [0.5, "#D45F5F"],
                [1, COLORS["primary"]],
            ],
            showscale=False,
        ),
        hovertemplate=(
            "<b>%{y}</b><br>"
            f"P(cascade within {window_hours:.0f}h): %{{x:.1%}}"
            "<extra></extra>"
        ),
    ))

    fig.update_layout(
        **LAYOUT_BASE,
        title=f"Top {top_n} Cascade Pairs (within {window_hours:.0f}h window)",
        xaxis=dict(
            **AXIS_STYLE,
            tickformat=".0%",
            title="Conditional Probability",
        ),
        yaxis=dict(**AXIS_STYLE),
        height=max(400, top_n * 30),
    )
    return fig


def plot_secondary_strike_curve(
    curve_df: pd.DataFrame,
    trigger_region: str,
) -> go.Figure:
    """
    Line chart showing P(any secondary alert | trigger in region) vs time window.

    Args:
        curve_df: Output of compute_secondary_strike_curve().
        trigger_region: Name of trigger region (for title).

    Returns:
        Plotly line figure.
    """
    region_short = trigger_region.replace(" oblast", "")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=curve_df["window_hours"],
        y=curve_df["probability"],
        mode="lines+markers",
        line=dict(color=COLORS["primary"], width=2.5),
        marker=dict(color=COLORS["primary"], size=8),
        fill="tozeroy",
        fillcolor="rgba(230, 57, 70, 0.1)",
        hovertemplate="Within <b>%{x}h</b>: P(secondary strike) = <b>%{y:.1%}</b><extra></extra>",
    ))

    fig.update_layout(
        **LAYOUT_BASE,
        title=f"National Secondary Strike Probability After Alert in {region_short}",
        xaxis=dict(**AXIS_STYLE, title="Time Window (hours)"),
        yaxis=dict(
            **AXIS_STYLE,
            tickformat=".0%",
            title="P(at least one secondary alert nationally)",
        ),
    )
    return fig
