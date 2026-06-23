"""
Project-wide constants for Ukraine air raid alert analysis.
"""

# Ukraine timezone (Kyiv)
KYIV_TZ = "Europe/Kyiv"

# Cascade analysis: max time window (hours) to consider a "related" alert
CASCADE_WINDOW_HOURS = 4

# Duration imputation: if end_time is missing, we cannot exceed this cap (hours)
MAX_IMPUTED_DURATION_HOURS = 6

# Overlap merge threshold (minutes): overlaps below this are merged
OVERLAP_MERGE_THRESHOLD_MINUTES = 5

# HMM: number of hidden regimes (low / medium / high threat)
HMM_N_COMPONENTS = 3

# Minimum data points required to fit a model for a given region
MIN_REGION_OBSERVATIONS = 30

# Oblast names (Ukrainian administrative regions, official transliteration)
OBLASTS: list[str] = [
    "Cherkasy",
    "Chernihiv",
    "Chernivtsi",
    "Crimea",
    "Dnipropetrovsk",
    "Donetsk",
    "Ivano-Frankivsk",
    "Kharkiv",
    "Kherson",
    "Khmelnytskyi",
    "Kirovohrad",
    "Kyiv",
    "Kyiv City",
    "Luhansk",
    "Lviv",
    "Mykolaiv",
    "Odesa",
    "Poltava",
    "Rivne",
    "Sumy",
    "Ternopil",
    "Vinnytsia",
    "Volyn",
    "Zakarpattia",
    "Zaporizhzhia",
    "Zhytomyr",
]

# Expected raw data columns (validated on load)
# volunteer_data_en.csv actual schema: region, started_at, finished_at, naive
EXPECTED_RAW_COLUMNS: list[str] = ["region", "started_at", "finished_at", "naive"]

# Processed DataFrame column names (canonical)
COL_REGION = "region"
COL_REGION_ID = "region_id"
COL_START = "started_at"
COL_END = "finished_at"  # matches volunteer_data_en.csv source column
COL_NAIVE = "naive"      # True = end_time was imputed by the source (not us)
COL_DURATION_MIN = "duration_minutes"
COL_IS_IMPUTED = "is_end_imputed"
COL_IS_OVERLAPPING = "is_overlapping"
COL_DATE = "date"
COL_HOUR = "hour"
COL_DOW = "day_of_week"
