"""
Forecasting module.

Wraps Prophet with ruptures-based structural changepoint injection.

Key design decisions (approved in plan review):
    - 14-day forecast horizon (wartime noise makes longer forecasts unreliable)
    - ruptures PELT algorithm detects structural breaks in alert volume
      (e.g., Oct 2022 infrastructure bombing campaign onset) and injects
      them into Prophet as explicit changepoints
    - We suppress Prophet's automatic changepoints (changepoint_prior_scale=0.01)
      to avoid overfitting to short-term fluctuations

Usage:
    from ukraine_alerts.models.forecasting import fit_prophet, plot_prophet_forecast
    forecast_df, model = fit_prophet(daily)
    fig = plot_prophet_forecast(forecast_df, daily, region="Kyivska oblast")
"""

from __future__ import annotations

import logging
import warnings

import pandas as pd
import plotly.graph_objects as go

from ukraine_alerts.eda.temporal import AXIS_STYLE, COLORS, LAYOUT_BASE

logger = logging.getLogger(__name__)

FORECAST_HORIZON_DAYS = 14


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_changepoints(
    daily: pd.DataFrame,
    n_breakpoints: int = 5,
    column: str = "alert_count",
) -> list[str]:
    """
    Use ruptures Binseg algorithm to detect structural breakpoints in the series.

    Binseg (Binary Segmentation) is used here instead of PELT because it accepts
    a fixed `n_bkps` parameter, which is easier to tune for this context.
    PELT requires a penalty value (unbounded search) which is harder to set
    without cross-validation on this kind of wartime data.

    Args:
        daily: Output of discretization.build_daily_series().
        n_breakpoints: Number of breakpoints to detect.
        column: Feature column to run changepoint detection on.

    Returns:
        List of ISO date strings for the detected changepoints.
    """
    try:
        import ruptures as rpt
    except ImportError as e:
        raise ImportError("ruptures is required: uv add ruptures") from e

    signal = daily[column].values.reshape(-1, 1)

    # Binseg: classic binary segmentation, O(n log n), supports n_bkps
    algo = rpt.Binseg(model="rbf", min_size=14, jump=1).fit(signal)
    # predict() returns sorted breakpoints; last element is len(signal) (sentinel)
    breakpoints = algo.predict(n_bkps=min(n_breakpoints, len(daily) // 14 - 1))

    dates = daily["ds"].values
    changepoint_dates = []
    for bp in breakpoints[:-1]:  # exclude terminal sentinel
        if 0 < bp < len(dates):
            changepoint_dates.append(str(dates[bp])[:10])

    logger.info("Detected %d structural changepoints", len(changepoint_dates))
    return changepoint_dates


def fit_prophet(
    daily: pd.DataFrame,
    horizon_days: int = FORECAST_HORIZON_DAYS,
    n_changepoints: int = 5,
) -> tuple[pd.DataFrame, object]:
    """
    Fit a Prophet model on daily alert counts with ruptures-detected changepoints.

    The model is designed for wartime data:
        - changepoint_prior_scale=0.01 (resist short-term noise)
        - Explicit changepoints from ruptures (structural campaign shifts)
        - weekly_seasonality=True, daily_seasonality=False, yearly_seasonality=True

    Args:
        daily: Output of discretization.build_daily_series().
        horizon_days: Days to forecast into the future.
        n_changepoints: Number of structural breaks to inject.

    Returns:
        (forecast_df, model): Prophet forecast DataFrame and fitted model object.
    """
    try:
        from prophet import Prophet
    except ImportError as e:
        raise ImportError("prophet is required: uv add prophet") from e

    # Detect structural changepoints
    changepoint_dates = detect_changepoints(daily, n_breakpoints=n_changepoints)

    prophet_df = daily[["ds", "alert_count"]].rename(columns={"alert_count": "y"}).copy()
    prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = Prophet(
            changepoints=changepoint_dates if changepoint_dates else None,
            changepoint_prior_scale=0.01,
            weekly_seasonality=True,
            daily_seasonality=False,
            yearly_seasonality=True,
            uncertainty_samples=300,
        )
        model.fit(prophet_df)

    future = model.make_future_dataframe(periods=horizon_days)
    forecast = model.predict(future)

    logger.info(
        "Prophet fitted and forecasted %d days ahead. Final yhat=%.2f",
        horizon_days,
        forecast["yhat"].iloc[-1],
    )
    return forecast, model


def plot_prophet_forecast(
    forecast: pd.DataFrame,
    actual: pd.DataFrame,
    region: str = "",
) -> go.Figure:
    """
    Plot Prophet forecast with uncertainty bands and actual alert count overlay.

    Args:
        forecast: Output of fit_prophet() — Prophet's forecast DataFrame.
        actual: Input daily DataFrame (for actual values overlay).
        region: Region name for the title.

    Returns:
        Plotly figure.
    """
    fig = go.Figure()

    # Uncertainty band (yhat_lower → yhat_upper)
    fig.add_trace(go.Scatter(
        x=pd.concat([forecast["ds"], forecast["ds"].iloc[::-1]]),
        y=pd.concat([forecast["yhat_upper"], forecast["yhat_lower"].iloc[::-1]]),
        fill="toself",
        fillcolor="rgba(69, 123, 157, 0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        name="80% Confidence",
        hoverinfo="skip",
    ))

    # Forecast line
    fig.add_trace(go.Scatter(
        x=forecast["ds"],
        y=forecast["yhat"].clip(lower=0),
        name="Prophet Forecast",
        mode="lines",
        line=dict(color=COLORS["accent"], width=2, dash="dot"),
        hovertemplate="<b>%{x}</b><br>Forecast: %{y:.1f}<extra></extra>",
    ))

    # Actual observations
    cutoff = actual["ds"].max()
    fig.add_trace(go.Scatter(
        x=actual["ds"],
        y=actual["alert_count"],
        name="Actual",
        mode="lines",
        line=dict(color=COLORS["text"], width=1.5),
        hovertemplate="<b>%{x}</b><br>Actual: %{y}<extra></extra>",
    ))

    # Vertical line at forecast start
    fig.add_vline(
        x=str(cutoff)[:10],
        line=dict(color=COLORS["primary"], width=1.5, dash="dash"),
        annotation_text="Forecast start",
        annotation_font_color=COLORS["text_muted"],
    )

    title = f"14-Day Alert Count Forecast — {region}" if region else "14-Day Alert Count Forecast"
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
