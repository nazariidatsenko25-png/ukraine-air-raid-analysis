"""
Cascade Analysis Page — cross-region conditional probability matrix.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ukraine_alerts.charts.cascade_charts import (
    plot_cascade_heatmap,
    plot_secondary_strike_curve,
    plot_top_cascade_pairs,
)
from ukraine_alerts.eda.cascade import (
    compute_cascade_matrix,
    compute_secondary_strike_curve,
)
from ukraine_alerts.ingestion import fetch_raw_data
from ukraine_alerts.preprocessing import build_clean_dataset
from ukraine_alerts.utils.constants import COL_REGION

st.set_page_config(page_title="Cascade — Air Raid Analysis", layout="wide", page_icon="🌪️")


@st.cache_data(show_spinner="Loading data…")
def load_data() -> pd.DataFrame:
    return build_clean_dataset(fetch_raw_data())


@st.cache_data(show_spinner="Computing cascade matrix (this takes ~15s)…")
def get_matrix(window_hours: float) -> pd.DataFrame:
    df = load_data()
    return compute_cascade_matrix(df, window_hours=window_hours)


@st.cache_data(show_spinner="Computing secondary strike curve…")
def get_strike_curve(region: str) -> pd.DataFrame:
    df = load_data()
    return compute_secondary_strike_curve(
        df, region, windows_hours=[0.5, 1, 2, 3, 4, 6, 8, 12]
    )


df = load_data()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.header("Parameters")
window_hours = st.sidebar.slider(
    "Cascade window (hours)",
    min_value=1,
    max_value=12,
    value=3,
    step=1,
    help="P(alert in region B within N hours | alert just started in region A)",
)
top_n_pairs = st.sidebar.slider("Top N pairs (bar chart)", 10, 25, 15, step=5)

st.sidebar.markdown("---")
st.sidebar.subheader("Secondary Strike")
regions = sorted(df[COL_REGION].unique().tolist())
default_idx = next(
    (i for i, r in enumerate(regions) if "Kyiv" in r and "City" not in r), 0
)
trigger_region = st.sidebar.selectbox("Trigger region", regions, index=default_idx)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

st.title("🌪️ Cascade Analysis")
st.markdown(
    "**Methodology**: For each alert in region A (the trigger), we scan the next "
    "`N` hours for alerts in all other regions. "
    "`P(B | A)` = fraction of A-triggers followed by ≥1 alert in B within the window."
)
st.info(
    "ℹ️ The cascade matrix recomputes when you change the window. "
    "First load takes ~15 seconds for 100k events.",
    icon="⏱",
)
st.markdown("---")

matrix = get_matrix(window_hours)

h1, h2 = st.columns([3, 2])
with h1:
    st.plotly_chart(
        plot_cascade_heatmap(
            matrix,
            title=f"Cross-Region Cascade Probability (window = {window_hours}h)",
        ),
        use_container_width=True,
    )
with h2:
    st.plotly_chart(
        plot_top_cascade_pairs(matrix, top_n=top_n_pairs, window_hours=window_hours),
        use_container_width=True,
    )

st.markdown("---")
st.subheader("Secondary Strike Probability Curve")
st.markdown(
    f"**P(any national alert within N hours | trigger in {trigger_region})**  \n"
    "Assumes conditional independence across regions (interpretable approximation)."
)

curve = get_strike_curve(trigger_region)
st.plotly_chart(
    plot_secondary_strike_curve(curve, trigger_region),
    use_container_width=True,
)

# Key stat callout
p_1h = curve.loc[curve["window_hours"] == 1, "probability"]
if not p_1h.empty:
    st.metric(
        label=f"P(national secondary alert within 1h | alert in {trigger_region.replace(' oblast', '')})",
        value=f"{p_1h.values[0]:.1%}",
    )
