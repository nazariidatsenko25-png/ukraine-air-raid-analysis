"""
Threat Profiles Page — Unsupervised classification of attack waves using GMM.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ukraine_alerts.charts.threat_charts import plot_threat_scatter, plot_threat_timeline
from ukraine_alerts.ingestion import fetch_raw_data
from ukraine_alerts.models.threat_clustering import fit_threat_gmm, group_attack_waves
from ukraine_alerts.preprocessing import build_clean_dataset

st.set_page_config(page_title="Threats — Air Raid Analysis", layout="wide", page_icon="🎯")


@st.cache_data(show_spinner="Loading data…")
def load_data() -> pd.DataFrame:
    return build_clean_dataset(fetch_raw_data())


@st.cache_data(show_spinner="Clustering attack waves (this takes ~10s)…")
def get_threat_clusters() -> pd.DataFrame:
    df = load_data()
    waves = group_attack_waves(df)
    labeled_waves, _, _ = fit_threat_gmm(waves)
    return labeled_waves


df = load_data()
waves = get_threat_clusters()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.header("About")
st.sidebar.markdown(
    "Since the raw dataset lacks labels for weapon systems, this page uses "
    "**Unsupervised Machine Learning (GMM)** to infer the threat profile of "
    "each attack wave based purely on its physical signature: duration and "
    "geographic spread."
)

st.sidebar.markdown("---")
st.sidebar.subheader("Wave-Breaker Logic")
st.sidebar.caption(
    "Alerts are grouped into continuous waves if they start within 30 minutes "
    "of each other. However, if a sudden spike occurs (≥5 regions in 3 mins), "
    "the wave is explicitly severed. This prevents slow drone swarms from "
    "chaining into sudden ballistic strikes."
)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

st.title("🎯 Threat Classification Profiles")
st.markdown(
    "Unsupervised discovery of adversary tactics. The GMM clusters overlapping "
    "regional alerts into physical signatures representing likely weapon classes."
)
st.markdown("---")

# ---------------------------------------------------------------------------
# 1. Physical Validation Scatter Plot
# ---------------------------------------------------------------------------

st.subheader("Physical Signature Validation")
st.markdown(
    "Does the math map to reality? We expect 3 distinct physical shapes: "
    "fast/wide (MiG/Ballistic), slow/wide (Shahed), and fast/localized (Artillery/Tactical)."
)

st.plotly_chart(plot_threat_scatter(waves), use_container_width=True)

st.markdown("---")

# ---------------------------------------------------------------------------
# 2. Tactical Shift Timeline
# ---------------------------------------------------------------------------

st.subheader("Evolution of Tactics")
st.markdown(
    "How has the composition of attacks changed over the course of the war? "
    "(Note: The Shahed cluster naturally emerges strongly after October 2022)."
)

st.plotly_chart(plot_threat_timeline(waves), use_container_width=True)

# ---------------------------------------------------------------------------
# 3. KPI Summary
# ---------------------------------------------------------------------------

st.markdown("---")
st.subheader("Profile Summaries")

c1, c2, c3 = st.columns(3)

for col, threat in zip(
    [c1, c2, c3],
    ["Tactical/Artillery", "Strategic/Ballistic (MiG)", "Loitering Munition (Shahed)"],
    strict=True,
):
    with col:
        subset = waves[waves["threat_profile"] == threat]
        if not subset.empty:
            st.markdown(f"### {threat}")
            st.metric("Total Waves", f"{len(subset):,}")
            st.metric("Median Duration", f"{subset['duration_min'].median():.0f} min")
            st.metric("Median Regions", f"{subset['region_count'].median():.0f}")
