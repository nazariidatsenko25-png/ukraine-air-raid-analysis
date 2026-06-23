"""
Modeling Page — HMM regime detection and Prophet forecasting.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ukraine_alerts.ingestion import fetch_raw_data
from ukraine_alerts.models.discretization import build_daily_series, list_regions_with_data
from ukraine_alerts.models.forecasting import fit_prophet, plot_prophet_forecast
from ukraine_alerts.models.hmm import decode_regimes, fit_hmm, plot_regime_overlay
from ukraine_alerts.preprocessing import build_clean_dataset

st.set_page_config(page_title="Modeling — Air Raid Analysis", layout="wide", page_icon="🔮")


@st.cache_data(show_spinner="Loading data…")
def load_data() -> pd.DataFrame:
    return build_clean_dataset(fetch_raw_data())


@st.cache_data(show_spinner="Building daily time series…")
def get_daily(region: str) -> pd.DataFrame:
    return build_daily_series(load_data(), region)


@st.cache_data(show_spinner="Fitting HMM (Viterbi decode)…")
def get_hmm_decoded(region: str) -> pd.DataFrame:
    daily = get_daily(region)
    model, scaler = fit_hmm(daily)
    return decode_regimes(daily, model, scaler)


@st.cache_data(show_spinner="Fitting Prophet (this takes ~20s)…")
def get_forecast(region: str) -> tuple[pd.DataFrame, object]:
    daily = get_daily(region)
    return fit_prophet(daily, horizon_days=14)


df = load_data()
modelable_regions = list_regions_with_data(df, min_days=30)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.header("Region")
default_idx = next(
    (i for i, r in enumerate(modelable_regions) if "Kyiv" in r and "City" not in r), 0
)
selected_region = st.sidebar.selectbox("Select region", modelable_regions, index=default_idx)

st.sidebar.markdown("---")
st.sidebar.caption(
    f"Only regions with ≥30 days of alerts are listed ({len(modelable_regions)} of {df['region'].nunique()})."
)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

st.title("🔮 Regime Detection & Forecasting")
st.markdown(f"**Region:** `{selected_region}`")
st.markdown("---")

# ---- HMM section ----
st.subheader("Threat Regime Detection (HMM)")
st.markdown(
    "A **Gaussian HMM** with 3 hidden states is fitted individually per region on two daily features: "
    "`alert_count` and `avg_duration_min`. States are decoded via the Viterbi algorithm, then labeled "
    "by intensity: **Low Threat** (blue) / **Elevated Threat** (amber) / **Crisis** (red)."
)

try:
    decoded = get_hmm_decoded(selected_region)

    # Regime summary stats
    regime_counts = decoded["regime_label"].value_counts()
    total_days = len(decoded)

    r1, r2, r3 = st.columns(3)
    with r1:
        n = regime_counts.get("Low Threat", 0)
        st.metric("🔵 Low Threat", f"{n} days", delta=f"{n/total_days:.0%} of total")
    with r2:
        n = regime_counts.get("Elevated Threat", 0)
        st.metric("🟡 Elevated Threat", f"{n} days", delta=f"{n/total_days:.0%} of total")
    with r3:
        n = regime_counts.get("Crisis", 0)
        st.metric("🔴 Crisis", f"{n} days", delta=f"{n/total_days:.0%} of total")

    st.plotly_chart(
        plot_regime_overlay(decoded, region=selected_region),
        use_container_width=True,
    )

    # Current regime callout
    latest = decoded.iloc[-1]
    regime_emoji = {"Low Threat": "🔵", "Elevated Threat": "🟡", "Crisis": "🔴"}
    current_regime = latest["regime_label"]
    st.info(
        f"{regime_emoji.get(current_regime, '❓')} **Current decoded regime**: {current_regime}  \n"
        f"Most recent date: {str(latest['ds'])[:10]}  |  "
        f"Alerts: {int(latest['alert_count'])}  |  "
        f"Avg duration: {latest['avg_duration_min']:.0f} min",
    )

except Exception as e:
    st.error(f"HMM fitting failed for {selected_region}: {e}")

st.markdown("---")

# ---- Prophet section ----
st.subheader("14-Day Forecast (Prophet + ruptures changepoints)")
st.markdown(
    "**Prophet** is fitted with structural changepoints auto-detected by `ruptures` PELT. "
    "`changepoint_prior_scale=0.01` suppresses overfitting to short-term volatility. "
    "The shaded band is the 80% uncertainty interval."
)

st.warning(
    "⚠️ Wartime data is inherently non-stationary. Prophet extrapolates recent campaign "
    "intensity — treat as a baseline, not an operational prediction.",
    icon="⚠️",
)

try:
    with st.spinner("Fitting Prophet (first run ~20s, then cached)…"):
        forecast, _ = get_forecast(selected_region)
    daily = get_daily(selected_region)

    st.plotly_chart(
        plot_prophet_forecast(forecast, daily, region=selected_region),
        use_container_width=True,
    )

    # 14-day forecast summary
    future_forecast = forecast[forecast["ds"] > daily["ds"].max()].head(14)
    f1, f2, f3 = st.columns(3)
    with f1:
        st.metric("Forecast avg (next 14d)", f"{future_forecast['yhat'].mean():.1f} alerts/day")
    with f2:
        st.metric("Forecast peak (next 14d)", f"{future_forecast['yhat'].max():.1f} alerts/day")
    with f3:
        st.metric(
            "Uncertainty range (avg)",
            f"±{((future_forecast['yhat_upper'] - future_forecast['yhat_lower']) / 2).mean():.1f}",
        )

except Exception as e:
    st.error(f"Prophet fitting failed for {selected_region}: {e}")
