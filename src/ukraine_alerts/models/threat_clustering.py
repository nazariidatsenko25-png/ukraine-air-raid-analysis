"""
Threat Clustering module.

Groups overlapping alerts into discrete 'Attack Waves' and uses a Gaussian
Mixture Model (GMM) to infer the physical weapon system profile (e.g.,
MiG-31K vs Shahed) based on geographic spread and temporal duration.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from ukraine_alerts.utils.constants import COL_REGION, COL_START

logger = logging.getLogger(__name__)

# Training cutoff approved in plan review (Shahed campaign began ~Oct 2022)
TRAINING_CUTOFF_DATE = pd.to_datetime("2022-10-01", utc=True)


def group_attack_waves(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group raw alerts into discrete national attack waves.

    Logic:
        1. Alerts are sorted chronologically.
        2. Continuity rule: An alert joins the current wave if it starts within
           30 minutes of the current wave's active window.
        3. Wave-breaker rule (Derivative check): If >= 5 regions alert within
           a 3-minute window, this indicates a new synchronized attack (e.g.,
           ballistic/MiG). If this happens *during* an existing slow wave
           (e.g., Shahed), the wave is severed and a new one begins.

    Args:
        df: Clean alert DataFrame.

    Returns:
        DataFrame of unique waves:
            wave_id, started_at, finished_at, duration_min, region_count
    """
    df = df.copy()
    
    # Exclude permanent sirens, otherwise their multi-year durations
    # will absorb every single other alert into a single infinite wave.
    permanent_regions = ["Luhanska oblast", "Autonomous Republic of Crimea"]
    df = df[~df[COL_REGION].isin(permanent_regions)]
    
    # Exclude extreme frontline artillery holds (>12 hours). A single 7-day
    # artillery alert in Nikopol will artificially stretch the duration of
    # whatever national wave it happens to coincide with.
    df = df[df["duration_minutes"] <= 12 * 60]
    
    df = df.sort_values(COL_START).reset_index(drop=True)

    if df.empty:
        return pd.DataFrame()

    # Calculate rolling 3-minute alert density to detect spikes (wave breakers)
    # We set index to started_at for rolling, then restore.
    df_temp = df.set_index(COL_START)
    df["rolling_3m_count"] = df_temp.index.to_series().rolling("3min").count().values

    # Pre-extract arrays for fast iteration
    starts = df[COL_START].values
    rolling_counts = df["rolling_3m_count"].values

    wave_ids = np.zeros(len(df), dtype=int)
    
    current_wave_id = 0
    prev_start_time = starts[0]
    thirty_mins = np.timedelta64(30, "m")

    # State for wave breaker:
    in_spike = rolling_counts[0] >= 5

    wave_ids[0] = current_wave_id

    for i in range(1, len(df)):
        start_time = starts[i]
        is_spike = rolling_counts[i] >= 5

        # Continuity: does this alert start within 30 minutes of the PREVIOUS alert starting?
        time_gap_exceeded = start_time > (prev_start_time + thirty_mins)
        wave_breaker_triggered = is_spike and not in_spike

        if time_gap_exceeded or wave_breaker_triggered:
            # Start new wave
            current_wave_id += 1
            
        wave_ids[i] = current_wave_id
        prev_start_time = start_time
        in_spike = is_spike

    df["wave_id"] = wave_ids

    # Aggregate waves
    waves = (
        df.groupby("wave_id")
        .agg(
            started_at=(COL_START, "min"),
            finished_at=("finished_at", "max"),
            region_count=(COL_REGION, "nunique"),
        )
        .reset_index(drop=True)
    )
    
    # Calculate duration
    waves["duration_min"] = (
        (waves["finished_at"] - waves["started_at"]).dt.total_seconds() / 60.0
    )
    
    logger.info("Grouped %d alerts into %d attack waves", len(df), len(waves))
    return waves


def fit_threat_gmm(waves: pd.DataFrame) -> tuple[pd.DataFrame, object, StandardScaler]:
    """
    Fit a Gaussian Mixture Model (n=3) on [duration_min, region_count].
    Only trains on data from Oct 1, 2022 onwards.

    Args:
        waves: Output of group_attack_waves.

    Returns:
        (waves_labeled, model, scaler): The waves df with assigned threat labels,
        the fitted GMM, and the scaler.
    """
    if waves.empty:
        return waves, None, None

    features = ["duration_min", "region_count"]
    
    # Filter training data (Oct 2022 onwards)
    train_mask = waves["started_at"] >= TRAINING_CUTOFF_DATE
    train_df = waves[train_mask]
    
    if len(train_df) < 10:
        logger.warning("Not enough data to train GMM after cutoff date.")
        waves["threat_profile"] = "Unknown"
        return waves, None, None

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[features].values.astype(float))

    gmm = GaussianMixture(n_components=3, covariance_type="full", random_state=42)
    gmm.fit(X_train)

    # Predict on the entire dataset (including pre-Oct 2022 data)
    X_all = scaler.transform(waves[features].values.astype(float))
    cluster_ids = gmm.predict(X_all)
    
    # Analyze cluster centroids to map them to physical signatures
    # We look at the unscaled centroids
    centroids = scaler.inverse_transform(gmm.means_)
    
    label_map = {}
    for i in range(3):
        dur, regs = centroids[i]
        if dur < 120 and regs > 10:
            # Short duration, massive spread
            label_map[i] = "Strategic/Ballistic (MiG)"
        elif dur > 180:
            # Long duration
            label_map[i] = "Loitering Munition (Shahed)"
        else:
            # Fallback for the remaining cluster
            label_map[i] = "Tactical/Artillery"

    # Edge case: if heuristics failed to distinctly map all 3 due to data shapes,
    # fallback to generic ranking by duration.
    if len(set(label_map.values())) < 3:
        logger.warning("GMM heuristic labeling failed, falling back to duration ranking.")
        durations = {i: centroids[i][0] for i in range(3)}
        ranked = sorted(durations, key=durations.get)
        label_map = {
            ranked[0]: "Tactical/Artillery",
            ranked[1]: "Strategic/Ballistic (MiG)",
            ranked[2]: "Loitering Munition (Shahed)",
        }

    waves["gmm_cluster"] = cluster_ids
    waves["threat_profile"] = waves["gmm_cluster"].map(label_map)
    
    logger.info("GMM fitted. Centroid mapping: %s", label_map)
    return waves, gmm, scaler
