# 🇺🇦 Ukraine Air Raid Alert Analysis

Time series analysis and forecasting of Ukrainian air raid alert data, built as a rapid-prototype analytics dashboard. Applies domain-appropriate statistical methods (Hawkes process cascade analysis, Hidden Markov regime detection, Prophet baseline forecasting) to inherently erratic, event-driven defense data.

## Features

- **EDA Dashboard**: Temporal heatmaps, regional comparison, day/hour distributions
- **Cascade Analysis**: Conditional probability matrix — P(region B alert | region A alert within N hours)
- **Regime Detection**: HMM-based threat level classification (low/elevated/crisis) per region
- **Forecasting**: Prophet baseline with uncertainty bands, per-region selection

## Setup

### Prerequisites

- Python 3.12+
- [`uv`](https://github.com/astral-sh/uv) (recommended) or `pip`

### Install with `uv` (recommended)

```bash
git clone https://github.com/YOUR_USERNAME/ukraine-air-raid-analysis.git
cd ukraine-air-raid-analysis
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### Install with `pip`

```bash
pip install -r requirements.txt
```

### Data

Data is automatically downloaded from the [Vadimkin Ukrainian air raid sirens dataset](https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset) on first run. No Kaggle account or API key required.

```bash
# Optional: pre-download data
python -m ukraine_alerts.ingestion
```

## Run the Dashboard

```bash
streamlit run app/app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

## Run Tests

```bash
pytest tests/ -v
```

## Project Structure

```
ukraine-air-raid-analysis/
├── src/ukraine_alerts/
│   ├── ingestion.py        # Data download + validation
│   ├── preprocessing.py    # Cleaning, imputation, feature engineering
│   ├── eda/                # Visualization modules
│   ├── models/             # HMM, Hawkes, Prophet
│   └── utils/constants.py  # Shared constants
├── app/
│   ├── app.py              # Streamlit entrypoint
│   └── pages/              # Multi-page Streamlit pages
├── scripts/
│   └── verify_phase1.py    # Pipeline smoke test
├── tests/                  # pytest test suite
└── pyproject.toml          # Dependencies + tooling config
```

## Methodological Notes

Standard time series models (ARIMA, Holt-Winters) are inappropriate for this data. Ukrainian air raid alerts are:
- **Non-stationary**: Frontline shifts cause structural breaks
- **Self-exciting**: A strike in one region significantly increases near-term probability in neighboring regions
- **Event-driven**: Driven by geopolitical decisions, not calendar seasonality

We use:
- **Prophet** as an explainable baseline (changepoint detection is actually useful here)
- **Hawkes process** cascade analysis for the self-excitation signal
- **GaussianHMM** for latent regime identification

## Data Quality

The raw dataset (volunteer-collected API mirror) has missing `ended_at` values. We impute using a per-region **log-normal duration model** fitted on observed durations. Imputed rows are flagged with `is_end_imputed=True` throughout the pipeline.

## License

MIT
