"""
EDA Page — temporal and regional visualizations.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ukraine_alerts.eda.regional import (
    plot_region_alert_ranking,
    plot_region_duration_comparison,
    plot_region_treemap,
    plot_regional_intensity_over_time,
)
from ukraine_alerts.eda.temporal import (
    plot_daily_alert_counts,
    plot_duration_distribution,
    plot_hourly_heatmap,
    plot_monthly_alert_counts,
)
from ukraine_alerts.ingestion import fetch_raw_data
from ukraine_alerts.preprocessing import build_clean_dataset
from ukraine_alerts.utils.constants import COL_REGION

st.set_page_config(page_title="EDA — Air Raid Analysis", layout="wide", page_icon="📈")


@st.cache_data(show_spinner="Loading data…")
def load_data() -> pd.DataFrame:
    return build_clean_dataset(fetch_raw_data())


df = load_data()

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

st.sidebar.header("Filters")
regions = sorted(df[COL_REGION].unique().tolist())
selected_region = st.sidebar.selectbox(
    "Region",
    ["All Regions"] + regions,
    index=0,
)
region_arg = None if selected_region == "All Regions" else selected_region

rolling = st.sidebar.slider("Rolling average window (days)", 7, 30, 14, step=7)
max_dur = st.sidebar.slider("Max duration shown (minutes)", 60, 600, 300, step=60)
top_n = st.sidebar.slider("Top N regions (ranking chart)", 10, 25, 25, step=5)

st.sidebar.markdown("---")
st.sidebar.caption("Extreme outliers (Luhansk, Crimea permanent sirens) excluded from duration stats by default.")

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

st.title("📈 Exploratory Data Analysis")
st.markdown("---")

# --- Temporal section ---
st.subheader("Temporal Patterns")

t1, t2 = st.columns([2, 1])
with t1:
    st.plotly_chart(
        plot_daily_alert_counts(df, region=region_arg, rolling_window=rolling),
        use_container_width=True,
    )
with t2:
    st.plotly_chart(
        plot_monthly_alert_counts(df, region=region_arg),
        use_container_width=True,
    )

st.plotly_chart(
    plot_hourly_heatmap(df, region=region_arg),
    use_container_width=True,
)

st.plotly_chart(
    plot_duration_distribution(df, region=region_arg, max_duration_minutes=max_dur),
    use_container_width=True,
)

st.markdown("---")

# --- Regional section ---
st.subheader("Regional Comparison")

r1, r2 = st.columns(2)
with r1:
    st.plotly_chart(
        plot_region_alert_ranking(df, top_n=top_n),
        use_container_width=True,
    )
with r2:
    st.plotly_chart(
        plot_region_duration_comparison(df, exclude_permanent=True),
        use_container_width=True,
    )

st.plotly_chart(
    plot_regional_intensity_over_time(df, top_n_regions=6, freq="W"),
    use_container_width=True,
)

st.plotly_chart(
    plot_region_treemap(df),
    use_container_width=True,
)
