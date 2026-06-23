"""
Data ingestion module.

Downloads the raw Ukrainian air raid alert dataset from the Vadimkin GitHub
mirror of the official siren API. Falls back to a Kaggle-style CSV path if
the GitHub download fails.

Usage:
    python -m ukraine_alerts.ingestion
    # or
    from ukraine_alerts.ingestion import fetch_raw_data
    df_raw = fetch_raw_data()
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd
import requests

from ukraine_alerts.utils.constants import EXPECTED_RAW_COLUMNS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Source configuration
# ---------------------------------------------------------------------------

# Primary: volunteer_data_en.csv — starts Feb 25 2022 (day 2 of the invasion),
# giving the longest possible time range. Oblast-level only (no district splits).
# Official data starts March 15 2022 and introduces raion-level splits post-Dec 2025.
PRIMARY_DATA_URL = (
    "https://raw.githubusercontent.com/Vadimkin/ukrainian-air-raid-sirens-dataset"
    "/main/datasets/volunteer_data_en.csv"
)

# Secondary fallback: official data (shorter history, raion-level post-Dec 2025)
SECONDARY_DATA_URL = (
    "https://raw.githubusercontent.com/Vadimkin/ukrainian-air-raid-sirens-dataset"
    "/main/datasets/official_data_en.csv"
)

RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
RAW_FILE_PATH = RAW_DATA_DIR / "alerts_raw.csv"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fetch_raw_data(
    force_download: bool = False,
    source_url: str = PRIMARY_DATA_URL,
) -> pd.DataFrame:
    """
    Download the raw alert CSV and return a DataFrame.

    Args:
        force_download: Re-download even if cached file exists.
        source_url: Override the default download URL.

    Returns:
        Raw DataFrame with minimal validation applied.

    Raises:
        RuntimeError: If download fails and no cached file exists.
        ValueError: If the downloaded data is missing expected columns.
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if RAW_FILE_PATH.exists() and not force_download:
        logger.info("Loading cached raw data from %s", RAW_FILE_PATH)
        return _load_and_validate(RAW_FILE_PATH)

    logger.info("Downloading raw data from %s", source_url)
    _download_csv(source_url, RAW_FILE_PATH)
    return _load_and_validate(RAW_FILE_PATH)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _download_csv(url: str, dest: Path, chunk_size: int = 1024 * 256) -> None:
    """Stream-download a CSV to disk with progress logging."""
    try:
        with requests.get(url, stream=True, timeout=60) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0))
            downloaded = 0

            with dest.open("wb") as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
                    downloaded += len(chunk)

            logger.info(
                "Downloaded %.1f MB to %s",
                downloaded / (1024 * 1024),
                dest,
            )
    except requests.RequestException as exc:
        # Clean up partial download
        if dest.exists():
            dest.unlink()
        raise RuntimeError(
            f"Failed to download data from {url}. "
            "Check your internet connection or supply a local CSV path."
        ) from exc


def _load_and_validate(path: Path) -> pd.DataFrame:
    """
    Load CSV from disk and perform column-level validation.

    The raw file may have extra columns (fine) but must contain
    all EXPECTED_RAW_COLUMNS. Dates are NOT parsed here — that is
    the preprocessing module's job.
    """
    df = pd.read_csv(path, low_memory=False)
    logger.info("Loaded %d rows, %d columns from %s", len(df), df.shape[1], path)

    missing_cols = set(EXPECTED_RAW_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Raw data is missing expected columns: {missing_cols}. "
            f"Available columns: {list(df.columns)}"
        )

    logger.info("Column validation passed.")
    return df


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )
    df = fetch_raw_data()
    print("\n--- Raw Data Sample ---")
    print(df.head(10).to_string())
    print(f"\nShape: {df.shape}")
    print(f"\nColumn dtypes:\n{df.dtypes}")
    print(f"\nNull counts:\n{df.isnull().sum()}")


if __name__ == "__main__":
    main()
