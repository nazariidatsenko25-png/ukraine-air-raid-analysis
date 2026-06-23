from functools import lru_cache

import pandas as pd

from ukraine_alerts.ingestion import fetch_raw_data
from ukraine_alerts.preprocessing import build_clean_dataset


@lru_cache(maxsize=1)
def get_cleaned_data() -> pd.DataFrame:
    """Load and cache the cleaned dataset globally."""
    raw = fetch_raw_data()
    return build_clean_dataset(raw)
