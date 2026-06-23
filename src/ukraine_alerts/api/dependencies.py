import threading
from functools import lru_cache

import pandas as pd

from ukraine_alerts.ingestion import fetch_raw_data
from ukraine_alerts.preprocessing import build_clean_dataset

_data_lock = threading.Lock()

@lru_cache(maxsize=1)
def _get_cleaned_data_cached() -> pd.DataFrame:
    raw = fetch_raw_data()
    return build_clean_dataset(raw)

def get_cleaned_data() -> pd.DataFrame:
    """Load and cache the cleaned dataset globally, thread-safe on first load."""
    with _data_lock:
        return _get_cleaned_data_cached()
