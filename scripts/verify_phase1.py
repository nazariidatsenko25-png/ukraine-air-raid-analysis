"""
Phase 1 verification script.

Run this to confirm the ingestion + preprocessing pipeline is working correctly.
Expected output:
    - Row count (should be > 100,000)
    - Imputed count (should be > 0 for real data)
    - No negative durations
    - Sample of clean data

Usage:
    python scripts/verify_phase1.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Ensure src/ is on the path when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ukraine_alerts.ingestion import fetch_raw_data
from ukraine_alerts.preprocessing import build_clean_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def main() -> None:
    print("=" * 60)
    print("PHASE 1 VERIFICATION: Ingestion + Preprocessing")
    print("=" * 60)

    # Step 1: Fetch
    print("\n[1/3] Fetching raw data...")
    df_raw = fetch_raw_data()
    print(f"  Raw shape: {df_raw.shape}")
    print(f"  Columns: {list(df_raw.columns)}")

    # Step 2: Preprocess
    print("\n[2/3] Running preprocessing pipeline...")
    df_clean = build_clean_dataset(df_raw)

    # Step 3: Report
    print("\n[3/3] Quality Report:")
    print(f"  Clean rows:         {len(df_clean):,}")
    print(f"  Dropped rows:       {len(df_raw) - len(df_clean):,}")
    print(f"  Imputed end_time:   {df_clean['is_end_imputed'].sum():,}")
    print(f"  Overlapping alerts: {df_clean['is_overlapping'].sum():,}")
    print(f"  Unique regions:     {df_clean['region'].nunique()}")
    print(f"  Date range:         {df_clean['date'].min()} → {df_clean['date'].max()}")

    print("\n  Duration stats (minutes):")
    print(df_clean["duration_minutes"].describe().round(2).to_string())

    print("\n  Sample (5 rows):")
    print(
        df_clean[
            ["region", "started_at", "finished_at", "duration_minutes", "is_end_imputed"]
        ]
        .head(5)
        .to_string()
    )

    assert df_clean["duration_minutes"].min() > 0, "FAIL: negative durations found"
    assert df_clean["region"].notna().all(), "FAIL: null regions found"
    assert df_clean["started_at"].notna().all(), "FAIL: null start times found"
    print("\n✅ All assertions passed. Phase 1 complete.")


if __name__ == "__main__":
    main()
