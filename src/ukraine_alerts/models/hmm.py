"""
HMM Regime Detection module.

Fits a GaussianHMM to a daily time series for a single region and decodes
the hidden state (threat regime) timeline using the Viterbi algorithm.

Model design:
    - 3 hidden states: Low Threat, Elevated Threat, Crisis
    - 2D observation space: [alert_count, avg_duration_min]
      (captures both frequency AND duration, separating MIG-31K short raids
      from Shahed long campaigns)
    - Individual model per region (not pooled national) — approved in plan review

Usage:
    from ukraine_alerts.models.hmm import fit_hmm, decode_regimes, plot_regime_overlay
    model, scaler = fit_hmm(daily_df)
    decoded = decode_regimes(daily_df, model, scaler)
    fig = plot_regime_overlay(decoded)
"""

from __future__ import annotations

import logging

import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler

from ukraine_alerts.eda.temporal import AXIS_STYLE, COLORS, LAYOUT_BASE
from ukraine_alerts.utils.constants import HMM_N_COMPONENTS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# State labels and visuals — ordered Low → Elevated → Crisis
# State assignment is post-hoc: sorted by mean alert_count of each state.
# ---------------------------------------------------------------------------

STATE_LABELS = ["Low Threat", "Elevated Threat", "Crisis"]
STATE_COLORS = [
    "rgba(69, 123, 157, 0.25)",   # blue — Low
    "rgba(244, 162, 97, 0.30)",   # amber — Elevated
    "rgba(230, 57, 70, 0.35)",    # red — Crisis
]
STATE_LINE_COLORS = [COLORS["accent"], "#F4A261", COLORS["primary"]]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fit_hmm(
    daily: pd.DataFrame,
    n_components: int = HMM_N_COMPONENTS,
    n_iter: int = 50,
    random_state: int = 42,
) -> tuple:
    """
    Fit a GaussianHMM on 2D daily features: [alert_count, avg_duration_min].

    Args:
        daily: Output of discretization.build_daily_series().
        n_components: Number of hidden states (default 3).
        n_iter: Max EM iterations.
        random_state: Reproducibility seed.

    Returns:
        (model, scaler): Fitted GaussianHMM and StandardScaler used to normalize features.

    Raises:
        ImportError: If hmmlearn is not installed.
        ValueError: If daily has fewer than n_components * 10 rows.
    """
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError as e:
        raise ImportError("hmmlearn is required: uv add hmmlearn") from e

    if len(daily) < n_components * 10:
        raise ValueError(
            f"Too few observations ({len(daily)}) to fit {n_components}-state HMM. "
            f"Need at least {n_components * 10}."
        )

    features = daily[["alert_count", "avg_duration_min"]].values.astype(float)

    # Standardize — GaussianHMM is sensitive to feature scale
    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    model = GaussianHMM(
        n_components=n_components,
        covariance_type="full",
        n_iter=n_iter,
        random_state=random_state,
        verbose=False,
    )
    model.fit(X)
    logger.info(
        "HMM fitted: %d states, converged=%s, log-likelihood=%.2f",
        n_components,
        model.monitor_.converged,
        model.monitor_.history[-1] if model.monitor_.history else float("nan"),
    )
    return model, scaler


def decode_regimes(
    daily: pd.DataFrame,
    model,
    scaler: StandardScaler,
) -> pd.DataFrame:
    """
    Viterbi-decode the most likely state sequence and label regimes.

    The raw HMM state indices (0, 1, 2) are remapped to human labels by
    sorting states by their mean alert_count (ascending). State with lowest
    mean alert_count → "Low Threat", highest → "Crisis".

    Args:
        daily: Same daily DataFrame used to fit the model.
        model: Fitted GaussianHMM from fit_hmm().
        scaler: StandardScaler from fit_hmm().

    Returns:
        daily DataFrame with added columns:
            - hmm_state_raw  : raw integer state from Viterbi
            - hmm_state      : remapped state index (0=low, 2=crisis)
            - regime_label   : human-readable label
    """
    features = daily[["alert_count", "avg_duration_min"]].values.astype(float)
    X = scaler.transform(features)
    raw_states = model.predict(X)

    # Remap states by mean alert_count
    state_means = {
        s: daily["alert_count"].values[raw_states == s].mean()
        for s in range(model.n_components)
    }
    rank_order = sorted(state_means, key=state_means.get)  # low → high
    remap = {raw: rank for rank, raw in enumerate(rank_order)}

    df = daily.copy()
    df["hmm_state_raw"] = raw_states
    df["hmm_state"] = [remap[s] for s in raw_states]
    df["regime_label"] = [STATE_LABELS[s] for s in df["hmm_state"]]
    return df


def plot_regime_overlay(
    decoded: pd.DataFrame,
    region: str = "",
) -> go.Figure:
    """
    Plot daily alert count as a time series with HMM regime bands overlaid.

    Args:
        decoded: Output of decode_regimes().
        region: Region name for the title.

    Returns:
        Plotly figure with regime bands + alert count line.
    """
    fig = go.Figure()

    # Draw regime bands as filled vertical spans
    _add_regime_bands(fig, decoded)

    # Alert count line on top
    fig.add_trace(go.Scatter(
        x=decoded["ds"],
        y=decoded["alert_count"],
        name="Daily Alerts",
        mode="lines",
        line=dict(color=COLORS["text"], width=1.5),
        hovertemplate="<b>%{x}</b><br>Alerts: %{y}<extra></extra>",
    ))

    # Invisible traces for legend entries
    for i, label in enumerate(STATE_LABELS):
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="markers",
            marker=dict(size=12, color=STATE_LINE_COLORS[i], symbol="square"),
            name=label,
            showlegend=True,
        ))

    title = f"Threat Regime Detection (HMM) — {region}" if region else "Threat Regime Detection (HMM)"
    fig.update_layout(
        **LAYOUT_BASE,
        title=title,
        xaxis=dict(**AXIS_STYLE, title=""),
        yaxis=dict(**AXIS_STYLE, title="Daily Alert Count"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color=COLORS["text_muted"]),
        ),
        hovermode="x unified",
    )
    return fig


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _add_regime_bands(fig: go.Figure, decoded: pd.DataFrame) -> None:
    """Add contiguous regime background bands (avoids one shape per day)."""
    if decoded.empty:
        return

    state_col = decoded["hmm_state"].values
    dates = decoded["ds"].values

    # Walk through the series and emit a band for each contiguous block
    start_idx = 0
    current_state = state_col[0]

    for i in range(1, len(state_col)):
        if state_col[i] != current_state or i == len(state_col) - 1:
            end_idx = i if state_col[i] != current_state else i + 1
            fig.add_vrect(
                x0=str(dates[start_idx])[:10],
                x1=str(dates[min(end_idx, len(dates) - 1)])[:10],
                fillcolor=STATE_COLORS[current_state],
                line_width=0,
                layer="below",
            )
            start_idx = i
            current_state = state_col[i]
