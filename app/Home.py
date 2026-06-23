"""
Home page — Ukraine Air Raid Alert Analysis Dashboard.

Loads and caches data, displays national KPI cards, and links to sub-pages.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ukraine_alerts.ingestion import fetch_raw_data
from ukraine_alerts.preprocessing import build_clean_dataset
from ukraine_alerts.utils.constants import COL_DATE, COL_DURATION_MIN, COL_REGION

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="🇺🇦 Ukraine Air Raid Analysis",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Data loading (cached globally — all pages reuse this)
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner="Loading alert data…")
def load_data() -> pd.DataFrame:
    return build_clean_dataset(fetch_raw_data())


df = load_data()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("🇺🇦 Air Raid Analysis")
st.sidebar.markdown("Navigate using the pages below.")
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Data**: [Vadimkin Volunteer Dataset](https://github.com/vadimkin/ukrainian-air-raids-datasets)  \n"
    f"**Period**: {df[COL_DATE].min()} → {df[COL_DATE].max()}  \n"
    f"**Records**: {len(df):,}"
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("🚨 Ukraine Air Raid Alert Analysis")
st.markdown(
    "Time series analysis of air raid alerts across Ukraine (Feb 2022 — present). "
    "Applies HMM regime detection, cascade probability analysis, and Prophet forecasting "
    "to inherently non-stationary wartime event data."
)

st.markdown("---")

# ---------------------------------------------------------------------------
# National KPI cards
# ---------------------------------------------------------------------------

total_alerts = len(df)
total_regions = df[COL_REGION].nunique()
total_days = (pd.to_datetime(df[COL_DATE].max()) - pd.to_datetime(df[COL_DATE].min())).days
median_duration = df[COL_DURATION_MIN].median()
longest_alert = df[COL_DURATION_MIN].max()
busiest_region = df[COL_REGION].value_counts().idxmax()

col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    st.metric("Total Alerts", f"{total_alerts:,}")
with col2:
    st.metric("Regions Monitored", total_regions)
with col3:
    st.metric("Days of Data", f"{total_days:,}")
with col4:
    st.metric("Median Duration", f"{median_duration:.0f} min")
with col5:
    st.metric("Longest Alert", f"{longest_alert/60:.0f} hrs")
with col6:
    st.metric("Most Targeted Region", busiest_region.replace(" oblast", ""))

st.markdown("---")

# ---------------------------------------------------------------------------
# Navigation cards
# ---------------------------------------------------------------------------

st.subheader("Explore")
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("### 📈 EDA")
    st.markdown(
        "Daily alert counts, hourly attack heatmaps, regional rankings, "
        "duration distributions."
    )
    st.page_link("pages/1_EDA.py", label="Open EDA →", icon="📈")

with c2:
    st.markdown("### 🌪️ Cascade Analysis")
    st.markdown(
        "Cross-region conditional probability matrix: "
        "P(alert in region B | alert in region A within N hours)."
    )
    st.page_link("pages/2_Cascade.py", label="Open Cascade →", icon="🌪️")

with c3:
    st.markdown("### 🔮 Regime & Forecast")
    st.markdown(
        "HMM threat regime detection (Low / Elevated / Crisis) + "
        "14-day Prophet forecast with ruptures changepoints."
    )
    st.page_link("pages/3_Modeling.py", label="Open Modeling →", icon="🔮")
