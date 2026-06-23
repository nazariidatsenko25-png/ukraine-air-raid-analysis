"""
Tests for the preprocessing pipeline.

Run with: pytest tests/test_preprocessing.py -v
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from ukraine_alerts.preprocessing import (
    _compute_duration,
    _engineer_temporal_features,
    _flag_overlapping_alerts,
    _impute_missing_end_times,
    _parse_datetimes,
    _select_columns,
    build_clean_dataset,
)
from ukraine_alerts.utils.constants import (
    COL_DATE,
    COL_DURATION_MIN,
    COL_END,
    COL_HOUR,
    COL_IS_IMPUTED,
    COL_IS_OVERLAPPING,
    COL_REGION,
    COL_START,
)


def make_raw_df(**overrides) -> pd.DataFrame:
    """Factory: minimal valid raw DataFrame for testing."""
    base = {
        "id": [1, 2, 3],
        "region_id": [1, 1, 2],
        "region": ["Kyiv", "Kyiv", "Lviv"],
        "started_at": [
            "2023-01-01T10:00:00+00:00",
            "2023-01-01T14:00:00+00:00",
            "2023-01-01T11:00:00+00:00",
        ],
        "ended_at": [
            "2023-01-01T11:30:00+00:00",
            "2023-01-01T15:00:00+00:00",
            "2023-01-01T12:00:00+00:00",
        ],
    }
    base.update(overrides)
    return pd.DataFrame(base)


class TestSelectColumns:
    def test_selects_canonical_columns(self):
        df = make_raw_df()
        result = _select_columns(df)
        assert COL_REGION in result.columns
        assert COL_START in result.columns
        assert COL_END in result.columns
        assert COL_IS_IMPUTED in result.columns

    def test_extra_columns_are_dropped(self):
        df = make_raw_df()
        df["unexpected_col"] = "noise"
        result = _select_columns(df)
        assert "unexpected_col" not in result.columns


class TestParseDatetimes:
    def test_parses_to_utc(self):
        df = _select_columns(make_raw_df())
        result = _parse_datetimes(df)
        assert result[COL_START].dt.tz.zone == "UTC"
        assert result[COL_END].dt.tz.zone == "UTC"

    def test_drops_rows_with_invalid_start(self):
        raw = make_raw_df()
        raw.at[0, "started_at"] = "NOT_A_DATE"
        df = _select_columns(raw)
        result = _parse_datetimes(df)
        assert len(result) == 2  # one row dropped


class TestImputeMissingEndTimes:
    def test_imputes_missing_end_time(self):
        raw = make_raw_df()
        raw.at[2, "ended_at"] = None  # Lviv alert has no end_time
        df = _select_columns(raw)
        df = _parse_datetimes(df)
        result = _impute_missing_end_times(df)

        assert result[COL_END].notna().all()
        assert result.loc[2, COL_IS_IMPUTED] is True or result.loc[2, COL_IS_IMPUTED] == True

    def test_non_missing_rows_not_flagged(self):
        df = _select_columns(make_raw_df())
        df = _parse_datetimes(df)
        result = _impute_missing_end_times(df)
        assert result[COL_IS_IMPUTED].sum() == 0


class TestComputeDuration:
    def test_duration_is_positive(self):
        df = _select_columns(make_raw_df())
        df = _parse_datetimes(df)
        df = _impute_missing_end_times(df)
        result = _compute_duration(df)
        assert (result[COL_DURATION_MIN] > 0).all()

    def test_duration_values_correct(self):
        df = _select_columns(make_raw_df())
        df = _parse_datetimes(df)
        df = _impute_missing_end_times(df)
        result = _compute_duration(df)
        # First alert: 10:00 → 11:30 = 90 minutes
        assert result.loc[0, COL_DURATION_MIN] == pytest.approx(90.0, abs=0.1)


class TestFlagOverlappingAlerts:
    def test_non_overlapping_unchanged(self):
        df = _select_columns(make_raw_df())
        df = _parse_datetimes(df)
        df = _impute_missing_end_times(df)
        df = _compute_duration(df)
        result = _flag_overlapping_alerts(df)
        assert result[COL_IS_OVERLAPPING].sum() == 0

    def test_overlapping_alerts_flagged(self):
        raw = pd.DataFrame({
            "id": [1, 2],
            "region_id": [1, 1],
            "region": ["Kyiv", "Kyiv"],
            "started_at": ["2023-01-01T10:00:00+00:00", "2023-01-01T10:30:00+00:00"],
            "ended_at":   ["2023-01-01T11:30:00+00:00", "2023-01-01T12:00:00+00:00"],
        })
        df = _select_columns(raw)
        df = _parse_datetimes(df)
        df = _impute_missing_end_times(df)
        df = _compute_duration(df)
        result = _flag_overlapping_alerts(df)
        # Both alerts overlap by 60 minutes → flagged, not merged
        assert result[COL_IS_OVERLAPPING].sum() == 2


class TestEngineeredFeatures:
    def test_date_and_hour_columns_exist(self):
        df = _select_columns(make_raw_df())
        df = _parse_datetimes(df)
        df = _impute_missing_end_times(df)
        df = _compute_duration(df)
        df = _flag_overlapping_alerts(df)
        result = _engineer_temporal_features(df)
        assert COL_DATE in result.columns
        assert COL_HOUR in result.columns


class TestBuildCleanDataset:
    def test_full_pipeline_runs(self):
        df = make_raw_df()
        result = build_clean_dataset(df)
        assert len(result) > 0
        assert result[COL_DURATION_MIN].min() > 0
        assert result[COL_START].notna().all()
