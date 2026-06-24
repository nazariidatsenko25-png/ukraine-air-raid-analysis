from functools import lru_cache

from fastapi import APIRouter, Depends, Response, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd

from ukraine_alerts.api.dependencies import get_cleaned_data
from ukraine_alerts.models.discretization import build_daily_series, list_regions_with_data
from ukraine_alerts.models.forecasting import fit_prophet
from ukraine_alerts.models.hmm import decode_regimes, fit_hmm
from ukraine_alerts.charts.model_charts import plot_prophet_forecast, plot_regime_overlay

router = APIRouter()


@lru_cache(maxsize=50)
def _cached_regimes_json(region: str, data_hash: int) -> str:
    """Fit HMM and return Plotly JSON. Cached per region."""
    from ukraine_alerts.api.dependencies import get_cleaned_data as _gcd
    df = _gcd()
    daily = build_daily_series(df, region)
    model, scaler = fit_hmm(daily)
    decoded = decode_regimes(daily, model, scaler)
    fig = plot_regime_overlay(decoded, region=region)
    return fig.to_json()


@lru_cache(maxsize=50)
def _cached_forecast_json(region: str, data_hash: int) -> str:
    """Fit Prophet and return Plotly JSON. Cached per region."""
    from ukraine_alerts.api.dependencies import get_cleaned_data as _gcd
    df = _gcd()
    daily = build_daily_series(df, region)
    forecast, _ = fit_prophet(daily, horizon_days=14)
    fig = plot_prophet_forecast(forecast, daily, region=region)
    return fig.to_json()


def _df_hash(df: pd.DataFrame) -> int:
    """Stable hash based on df shape + last row — cheap, good enough."""
    return hash((len(df), str(df.iloc[-1].values.tolist()) if len(df) else 0))


@router.get("/regions")
def get_modelable_regions(df: pd.DataFrame = Depends(get_cleaned_data)):
    regions = list_regions_with_data(df, min_days=30)
    return JSONResponse(content={"regions": regions})


@router.get("/{region}/regimes")
def get_regimes(region: str, df: pd.DataFrame = Depends(get_cleaned_data)):
    regions = list_regions_with_data(df, min_days=30)
    if region not in regions:
        raise HTTPException(status_code=400, detail="Not enough data for this region")
    json_str = _cached_regimes_json(region, _df_hash(df))
    return Response(content=json_str, media_type="application/json")


@router.get("/{region}/forecast")
def get_forecast(region: str, df: pd.DataFrame = Depends(get_cleaned_data)):
    regions = list_regions_with_data(df, min_days=30)
    if region not in regions:
        raise HTTPException(status_code=400, detail="Not enough data for this region")
    json_str = _cached_forecast_json(region, _df_hash(df))
    return Response(content=json_str, media_type="application/json")
