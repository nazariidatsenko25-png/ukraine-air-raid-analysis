"""
Discretization module.

Converts the event-based alert DataFrame (one row = one alert) into a
regular daily time series suitable for HMM and Prophet.

The output is a per-region daily DataFrame with:
    - ds          : date (no gaps — zero-filled)
    - alert_count : number of alerts that started on that day
    - total_duration_min : sum of alert durations for that day
    - avg_duration_min   : mean alert duration (0 if no alerts)

Usage:
    from ukraine_alerts.models.discretization import build_daily_series, build_national_series
    daily = build_daily_series(df_clean, region="Kyivska oblast")
    national = build_national_series(df_clean)
"""

from __future__ import annotations

import pandas as pd

from ukraine_alerts.utils.constants import COL_DATE, COL_DURATION_MIN, COL_REGION

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_daily_series(
    df: pd.DataFrame,
    region: str,
) -> pd.DataFrame:
    """
    Build a zero-gap daily time series for a single region.

    Args:
        df: Clean alert DataFrame from preprocessing.build_clean_dataset().
        region: Region name to filter on.

    Returns:
        DataFrame indexed by date with columns:
            ds, alert_count, total_duration_min, avg_duration_min.
        Every date from data start to data end is present (gaps filled with 0).

    Raises:
        ValueError: If region has no data in df.
    """
    df_r = df[df[COL_REGION] == region].copy()
    if df_r.empty:
        raise ValueError(f"No data for region: {region!r}")

    df_r["ds"] = pd.to_datetime(df_r[COL_DATE])

    daily = (
        df_r.groupby("ds")
        .agg(
            alert_count=(COL_DURATION_MIN, "count"),
            total_duration_min=(COL_DURATION_MIN, "sum"),
            avg_duration_min=(COL_DURATION_MIN, "mean"),
        )
        .reset_index()
        .sort_values("ds")
    )

    # Fill date gaps with zeros (HMM and Prophet both require regular grids)
    full_range = pd.date_range(daily["ds"].min(), daily["ds"].max(), freq="D")
    daily = (
        daily.set_index("ds")
        .reindex(full_range)
        .fillna(0.0)
        .rename_axis("ds")
        .reset_index()
    )
    return daily


def build_national_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a zero-gap national daily time series (aggregated across all regions).

    Returns:
        DataFrame with columns: ds, alert_count, total_duration_min, avg_duration_min.
    """
    df = df.copy()
    df["ds"] = pd.to_datetime(df[COL_DATE])

    daily = (
        df.groupby("ds")
        .agg(
            alert_count=(COL_DURATION_MIN, "count"),
            total_duration_min=(COL_DURATION_MIN, "sum"),
            avg_duration_min=(COL_DURATION_MIN, "mean"),
        )
        .reset_index()
        .sort_values("ds")
    )

    full_range = pd.date_range(daily["ds"].min(), daily["ds"].max(), freq="D")
    daily = (
        daily.set_index("ds")
        .reindex(full_range)
        .fillna(0.0)
        .rename_axis("ds")
        .reset_index()
    )
    return daily


def list_regions_with_data(df: pd.DataFrame, min_days: int = 30) -> list[str]:
    """
    Return regions with at least `min_days` distinct alert days.
    Filters out regions with too few observations to model reliably.

    Args:
        df: Clean alert DataFrame.
        min_days: Minimum unique days with at least one alert.

    Returns:
        Sorted list of qualifying region names.
    """
    counts = (
        df.groupby(COL_REGION)[COL_DATE]
        .nunique()
        .reset_index(name="active_days")
    )
    return sorted(counts[counts["active_days"] >= min_days][COL_REGION].tolist())
