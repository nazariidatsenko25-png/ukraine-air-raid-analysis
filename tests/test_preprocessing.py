"""
Tests for the preprocessing pipeline.

Run with: pytest tests/test_preprocessing.py -v
"""

from __future__ import annotations

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

# ---------------------------------------------------------------------------
# Fixtures / Factories
# ---------------------------------------------------------------------------


def make_raw_df(**overrides) -> pd.DataFrame:
    """
    Factory: minimal valid raw DataFrame matching volunteer_data_en.csv schema.
    Columns: region, started_at, finished_at, naive
    """
    base = {
        "region": ["Kyiv", "Kyiv", "Lviv"],
        "started_at": [
            "2023-01-01T10:00:00+00:00",
            "2023-01-01T14:00:00+00:00",
            "2023-01-01T11:00:00+00:00",
        ],
        "finished_at": [
            "2023-01-01T11:30:00+00:00",
            "2023-01-01T15:00:00+00:00",
            "2023-01-01T12:00:00+00:00",
        ],
        "naive": [False, False, False],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def make_overlapping_df() -> pd.DataFrame:
    """Two Kyiv alerts overlapping by 60 minutes."""
    return pd.DataFrame({
        "region": ["Kyiv", "Kyiv"],
        "started_at": ["2023-01-01T10:00:00+00:00", "2023-01-01T10:30:00+00:00"],
        "finished_at": ["2023-01-01T11:30:00+00:00", "2023-01-01T12:00:00+00:00"],
        "naive": [False, False],
    })


# ---------------------------------------------------------------------------
# TestSelectColumns
# ---------------------------------------------------------------------------


class TestSelectColumns:
    def test_selects_canonical_columns(self):
        result = _select_columns(make_raw_df())
        assert COL_REGION in result.columns
        assert COL_START in result.columns
        assert COL_END in result.columns       # finished_at
        assert COL_IS_IMPUTED in result.columns

    def test_extra_columns_are_dropped(self):
        df = make_raw_df()
        df["unexpected_col"] = "noise"
        result = _select_columns(df)
        assert "unexpected_col" not in result.columns

    def test_naive_flag_preserved(self):
        result = _select_columns(make_raw_df())
        assert "naive" in result.columns
        assert result["naive"].dtype == bool

    def test_naive_defaults_to_false_when_absent(self):
        df = make_raw_df().drop(columns=["naive"])
        result = _select_columns(df)
        assert "naive" in result.columns
        assert result["naive"].all() is False or not result["naive"].any()


# ---------------------------------------------------------------------------
# TestParseDatetimes
# ---------------------------------------------------------------------------


class TestParseDatetimes:
    def test_parses_to_utc(self):
        df = _select_columns(make_raw_df())
        result = _parse_datetimes(df)
        # pandas 2.x: use str(tz) instead of .tz.zone (removed in 2.x)
        assert str(result[COL_START].dt.tz) == "UTC"
        assert str(result[COL_END].dt.tz) == "UTC"

    def test_drops_rows_with_invalid_start(self):
        raw = make_raw_df()
        raw.at[0, "started_at"] = "NOT_A_DATE"
        df = _select_columns(raw)
        result = _parse_datetimes(df)
        assert len(result) == 2  # one row dropped


# ---------------------------------------------------------------------------
# TestImputeMissingEndTimes
# ---------------------------------------------------------------------------


class TestImputeMissingEndTimes:
    def test_imputes_truly_missing_end_time(self):
        raw = make_raw_df()
        raw.at[2, "finished_at"] = None  # Lviv alert has no end_time, naive=False
        df = _select_columns(raw)
        df = _parse_datetimes(df)
        result = _impute_missing_end_times(df)

        assert result[COL_END].notna().all()
        assert result.loc[2, COL_IS_IMPUTED] == True  # noqa: E712

    def test_non_missing_rows_not_flagged(self):
        df = _select_columns(make_raw_df())
        df = _parse_datetimes(df)
        result = _impute_missing_end_times(df)
        assert result[COL_IS_IMPUTED].sum() == 0

    def test_naive_rows_flagged_as_imputed(self):
        raw = make_raw_df()
        raw.at[0, "naive"] = True  # source already imputed this
        df = _select_columns(raw)
        df = _parse_datetimes(df)
        result = _impute_missing_end_times(df)
        # The naive row should be flagged is_end_imputed=True
        assert result.loc[0, COL_IS_IMPUTED] == True  # noqa: E712
        # The non-naive rows should NOT be flagged
        assert result.loc[1, COL_IS_IMPUTED] == False  # noqa: E712


# ---------------------------------------------------------------------------
# TestComputeDuration
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# TestFlagOverlappingAlerts
# ---------------------------------------------------------------------------


class TestFlagOverlappingAlerts:
    def test_non_overlapping_unchanged(self):
        df = _select_columns(make_raw_df())
        df = _parse_datetimes(df)
        df = _impute_missing_end_times(df)
        df = _compute_duration(df)
        result = _flag_overlapping_alerts(df)
        assert result[COL_IS_OVERLAPPING].sum() == 0

    def test_overlapping_alerts_flagged(self):
        df = _select_columns(make_overlapping_df())
        df = _parse_datetimes(df)
        df = _impute_missing_end_times(df)
        df = _compute_duration(df)
        result = _flag_overlapping_alerts(df)
        # Both alerts overlap by 60 minutes → flagged, not merged
        assert result[COL_IS_OVERLAPPING].sum() == 2


# ---------------------------------------------------------------------------
# TestEngineeredFeatures
# ---------------------------------------------------------------------------


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

    def test_hour_in_kyiv_local_time(self):
        """Hour column must reflect Kyiv local time (UTC+2/UTC+3), not UTC."""
        df = _select_columns(make_raw_df())
        df = _parse_datetimes(df)
        df = _impute_missing_end_times(df)
        df = _compute_duration(df)
        df = _flag_overlapping_alerts(df)
        result = _engineer_temporal_features(df)
        # started_at = 2023-01-01T10:00:00+00:00 → Kyiv (UTC+2 in winter) = 12:00
        assert result.loc[0, COL_HOUR] == 12


# ---------------------------------------------------------------------------
# TestBuildCleanDataset (integration)
# ---------------------------------------------------------------------------


class TestBuildCleanDataset:
    def test_full_pipeline_runs(self):
        result = build_clean_dataset(make_raw_df())
        assert len(result) > 0
        assert result[COL_DURATION_MIN].min() > 0
        assert result[COL_START].notna().all()

    def test_output_has_all_expected_columns(self):
        result = build_clean_dataset(make_raw_df())
        required = {COL_REGION, COL_START, COL_END, COL_DURATION_MIN,
                    COL_IS_IMPUTED, COL_IS_OVERLAPPING, COL_DATE, COL_HOUR}
        assert required.issubset(set(result.columns))
