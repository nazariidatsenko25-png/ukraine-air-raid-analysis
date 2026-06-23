"""
Data preprocessing pipeline.

Transforms raw alert data into a clean, analytically-ready DataFrame.
Handles:
    - Datetime parsing with timezone normalization (UTC storage, Kyiv display)
    - Missing end_time imputation via region-specific log-normal duration model
    - Overlapping alert merging within the same region
    - Feature engineering (duration, date parts, flags)

Usage:
    from ukraine_alerts.preprocessing import build_clean_dataset
    df_clean = build_clean_dataset(df_raw)
"""

from __future__ import annotations

import logging
from typing import Final

import numpy as np
import pandas as pd

from ukraine_alerts.utils.constants import (
    COL_DATE,
    COL_DOW,
    COL_DURATION_MIN,
    COL_END,
    COL_HOUR,
    COL_IS_IMPUTED,
    COL_IS_OVERLAPPING,
    COL_NAIVE,
    COL_REGION,
    COL_START,
    KYIV_TZ,
    MAX_IMPUTED_DURATION_HOURS,
    OVERLAP_MERGE_THRESHOLD_MINUTES,
)

logger = logging.getLogger(__name__)

# Duration floor (minutes): alerts shorter than this are suspicious
MIN_VALID_DURATION_MINUTES: Final[int] = 1

# UTC timezone string for pandas
UTC: Final[str] = "UTC"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_clean_dataset(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Full preprocessing pipeline. Returns a clean DataFrame ready for EDA and modeling.

    Pipeline stages:
        1. Select & rename columns to canonical names
        2. Parse datetimes, normalize to UTC
        3. Impute missing end_time using per-region log-normal duration model
        4. Compute alert duration in minutes
        5. Flag and merge overlapping alerts in the same region
        6. Engineer temporal features (date, hour, day-of-week)
        7. Final sanity filter: drop rows with duration <= 0

    Args:
        df_raw: Output of ingestion.fetch_raw_data()

    Returns:
        Clean DataFrame with canonical column names and quality flags.
    """
    logger.info("Starting preprocessing pipeline on %d rows", len(df_raw))

    df = _select_columns(df_raw)
    df = _parse_datetimes(df)
    df = _impute_missing_end_times(df)
    df = _compute_duration(df)
    df = _flag_overlapping_alerts(df)
    df = _engineer_temporal_features(df)
    df = _final_filter(df)

    logger.info(
        "Preprocessing complete. Output: %d rows, %d imputed, %d overlapping",
        len(df),
        df[COL_IS_IMPUTED].sum(),
        df[COL_IS_OVERLAPPING].sum(),
    )
    return df


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------


def _select_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Select and rename to canonical column names."""
    # Capture naive BEFORE column filtering since it's not in col_map
    naive_series = df["naive"].astype(bool) if "naive" in df.columns else pd.Series([False] * len(df), index=df.index)

    col_map = {
        "region": COL_REGION,
        "started_at": COL_START,
        "finished_at": COL_END,  # volunteer_data_en.csv uses finished_at
    }
    available = {k: v for k, v in col_map.items() if k in df.columns}
    df = df[list(available.keys())].rename(columns=available).copy()

    # Preserve the 'naive' flag.
    # naive=True means the source already imputed finished_at (start + 30min).
    # We must NOT double-impute these rows.
    df[COL_NAIVE] = naive_series.values
    df[COL_IS_IMPUTED] = False
    df[COL_IS_OVERLAPPING] = False
    return df


def _parse_datetimes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse started_at and ended_at to UTC-aware datetimes.

    The Vadimkin dataset stores ISO 8601 strings with timezone info.
    We normalize everything to UTC; Kyiv-local conversion happens at display time.
    """
    for col in [COL_START, COL_END]:
        df[col] = pd.to_datetime(df[col], utc=True, format="ISO8601", errors="coerce")

    n_bad_start = df[COL_START].isna().sum()
    n_bad_end = df[COL_END].isna().sum()

    if n_bad_start > 0:
        logger.warning(
            "Dropping %d rows with unparseable started_at values", n_bad_start
        )
        df = df.dropna(subset=[COL_START])

    logger.info(
        "%d rows have missing ended_at (will be imputed)", n_bad_end
    )
    return df


def _impute_missing_end_times(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing end_time using a per-region log-normal duration model.

    Important: rows with naive=True already had their end_time imputed by the
    source (start + 30min). We flag them with is_end_imputed=True but do NOT
    re-impute them — the source's 30min default is already baked in.

    For rows where finished_at is truly NaN (different from naive), we apply
    our log-normal model.

    Strategy:
        1. Flag source-imputed (naive=True) rows as is_end_imputed=True.
        2. For each region, compute observed durations from non-naive rows.
        3. Fit log-normal parameters (mu, sigma) to observed durations.
        4. Impute truly missing end_time as exp(mu) (median of log-normal).
        5. Cap imputed duration at MAX_IMPUTED_DURATION_HOURS.
        6. Fall back to global median if a region has < 5 observed durations.
    """
    df = df.copy()

    # Mark source-imputed rows
    source_imputed_mask = df[COL_NAIVE].fillna(False).astype(bool)
    df.loc[source_imputed_mask, COL_IS_IMPUTED] = True

    # Find rows with truly missing finished_at (NaN, not source-imputed)
    truly_missing_mask = df[COL_END].isna() & ~source_imputed_mask
    if not truly_missing_mask.any():
        logger.info(
            "%d source-imputed rows flagged (naive=True). 0 truly missing end_times.",
            source_imputed_mask.sum(),
        )
        return df

    # Compute observed durations for non-naive, non-missing rows
    reliable_mask = ~source_imputed_mask & df[COL_END].notna()
    observed = df[reliable_mask].copy()
    observed["_obs_duration_min"] = (
        observed[COL_END] - observed[COL_START]
    ).dt.total_seconds() / 60.0
    observed = observed[observed["_obs_duration_min"] > MIN_VALID_DURATION_MINUTES]

    # Global fallback
    global_log_durations = np.log(observed["_obs_duration_min"].clip(lower=1.0))
    global_mu = float(global_log_durations.mean())
    global_median_min = float(np.exp(global_mu))

    # Per-region log-normal fit
    region_medians: dict[str, float] = {}
    for region, group in observed.groupby(COL_REGION):
        durations = group["_obs_duration_min"]
        if len(durations) >= 5:
            log_d = np.log(durations.clip(lower=1.0))
            region_medians[region] = float(np.exp(log_d.mean()))
        else:
            region_medians[region] = global_median_min

    cap_minutes = MAX_IMPUTED_DURATION_HOURS * 60

    def _get_imputed_duration(region: str) -> float:
        return min(region_medians.get(region, global_median_min), cap_minutes)

    missing_idx = df.index[truly_missing_mask]
    for idx in missing_idx:
        region = df.at[idx, COL_REGION]
        imputed_duration = _get_imputed_duration(region)
        raw_ts = df.at[idx, COL_START] + pd.Timedelta(minutes=imputed_duration)
        # Cast to the same resolution as the column (pandas 2.x uses datetime64[us])
        df.at[idx, COL_END] = raw_ts.as_unit("us")
        df.at[idx, COL_IS_IMPUTED] = True

    logger.info(
        "%d source-imputed (naive=True), %d truly missing end_time imputed by us",
        source_imputed_mask.sum(),
        truly_missing_mask.sum(),
    )
    return df


def _compute_duration(df: pd.DataFrame) -> pd.DataFrame:
    """Compute alert duration in minutes. Negative durations indicate data errors."""
    df = df.copy()
    df[COL_DURATION_MIN] = (
        (df[COL_END] - df[COL_START]).dt.total_seconds() / 60.0
    ).round(2)
    n_negative = (df[COL_DURATION_MIN] < 0).sum()
    if n_negative > 0:
        logger.warning(
            "%d rows have negative duration (end_time < start_time) — will be dropped in final filter",
            n_negative,
        )
    return df


def _flag_overlapping_alerts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect overlapping alerts within the same region.

    Two alerts in the same region overlap if alert B starts before alert A ends.
    If the overlap is < OVERLAP_MERGE_THRESHOLD_MINUTES, we merge them.
    Larger overlaps are flagged but kept (may be real data — e.g., extended alert).
    """
    df = df.sort_values([COL_REGION, COL_START]).copy()
    result_rows: list[pd.Series] = []

    for _region, group in df.groupby(COL_REGION, sort=False):
        group = group.reset_index(drop=True)
        merged: list[pd.Series] = []

        for _, row in group.iterrows():
            if not merged:
                merged.append(row)
                continue

            prev = merged[-1]
            # Check if current alert starts before previous ends
            if row[COL_START] < prev[COL_END]:
                overlap_minutes = (
                    prev[COL_END] - row[COL_START]
                ).total_seconds() / 60.0

                if overlap_minutes <= OVERLAP_MERGE_THRESHOLD_MINUTES:
                    # Merge: extend previous alert's end_time if needed
                    if row[COL_END] > prev[COL_END]:
                        merged[-1] = prev.copy()
                        merged[-1][COL_END] = row[COL_END]
                    # Skip the current row (absorbed into previous)
                    continue
                else:
                    # Real overlap: flag both
                    merged[-1][COL_IS_OVERLAPPING] = True
                    row = row.copy()
                    row[COL_IS_OVERLAPPING] = True

            merged.append(row)

        result_rows.extend(merged)

    df_clean = pd.DataFrame(result_rows).reset_index(drop=True)
    n_overlapping = df_clean[COL_IS_OVERLAPPING].sum()
    n_dropped = len(df) - len(df_clean)
    logger.info(
        "Overlap processing: %d merged (dropped), %d flagged as overlapping",
        n_dropped,
        n_overlapping,
    )
    return df_clean


def _engineer_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add date-part columns derived from started_at.

    All derived features use Kyiv local time for human-interpretable display,
    but the underlying datetime columns remain UTC.
    """
    df = df.copy()
    started_kyiv = df[COL_START].dt.tz_convert(KYIV_TZ)
    df[COL_DATE] = started_kyiv.dt.date
    df[COL_HOUR] = started_kyiv.dt.hour
    df[COL_DOW] = started_kyiv.dt.day_name()
    return df


def _final_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with non-positive duration (data errors that survived imputation)."""
    before = len(df)
    df = df[df[COL_DURATION_MIN] > MIN_VALID_DURATION_MINUTES].copy()
    dropped = before - len(df)
    if dropped > 0:
        logger.warning("Final filter dropped %d rows with invalid duration", dropped)
    return df.reset_index(drop=True)
