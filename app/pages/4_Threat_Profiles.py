"""
Threat Profiles Page — Unsupervised classification of attack waves using GMM.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ukraine_alerts.eda.temporal import AXIS_STYLE, COLORS, LAYOUT_BASE
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

# Define custom color map for the threat profiles
THREAT_COLORS = {
    "Strategic/Ballistic (MiG)": COLORS["accent"],  # Red
    "Loitering Munition (Shahed)": "#F4A261",       # Amber
    "Tactical/Artillery": COLORS["primary"],        # Blue
    "Unknown": COLORS["text_muted"],
}

fig_scatter = px.scatter(
    waves,
    x="duration_min",
    y="region_count",
    color="threat_profile",
    color_discrete_map=THREAT_COLORS,
    opacity=0.4,
    hover_data=["started_at"],
    title="Attack Waves by Duration and Geographic Spread",
)

fig_scatter.update_layout(
    **LAYOUT_BASE,
    xaxis=dict(
        **AXIS_STYLE,
        title="Wave Duration (minutes)",
        range=[0, 800],  # Cap visual x-axis to focus on the main clusters
    ),
    yaxis=dict(
        **AXIS_STYLE,
        title="Number of Regions Affected",
    ),
    legend_title_text="Inferred Threat Profile",
)
fig_scatter.update_traces(marker=dict(size=6, line=dict(width=0)))
st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown("---")

# ---------------------------------------------------------------------------
# 2. Tactical Shift Timeline
# ---------------------------------------------------------------------------

st.subheader("Evolution of Tactics")
st.markdown(
    "How has the composition of attacks changed over the course of the war? "
    "(Note: The Shahed cluster naturally emerges strongly after October 2022)."
)

# Resample waves to monthly counts by threat profile
waves["month"] = waves["started_at"].dt.to_period("M").dt.to_timestamp()
monthly_threats = (
    waves.groupby(["month", "threat_profile"])
    .size()
    .reset_index(name="count")
)

fig_bar = px.bar(
    monthly_threats,
    x="month",
    y="count",
    color="threat_profile",
    color_discrete_map=THREAT_COLORS,
    title="Monthly Attack Waves by Inferred Threat",
)

fig_bar.update_layout(
    **LAYOUT_BASE,
    xaxis=dict(**AXIS_STYLE, title=""),
    yaxis=dict(**AXIS_STYLE, title="Number of Attack Waves"),
    legend_title_text="",
    barmode="stack",
)
st.plotly_chart(fig_bar, use_container_width=True)

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
