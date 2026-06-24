from functools import lru_cache

from fastapi import APIRouter, Depends, Response
from fastapi.responses import JSONResponse
import pandas as pd

from ukraine_alerts.api.dependencies import get_cleaned_data
from ukraine_alerts.eda.cascade import compute_cascade_matrix, compute_secondary_strike_curve
from ukraine_alerts.charts.cascade_charts import plot_cascade_heatmap, plot_secondary_strike_curve

router = APIRouter()


def _df_hash(df: pd.DataFrame) -> int:
    return hash((len(df), str(df.iloc[-1].values.tolist()) if len(df) else 0))


@lru_cache(maxsize=1)
def _cached_cascade_matrix(data_hash: int) -> pd.DataFrame:
    from ukraine_alerts.api.dependencies import get_cleaned_data as _gcd
    df = _gcd()
    return compute_cascade_matrix(df, window_hours=3)


@lru_cache(maxsize=1)
def _cached_heatmap_json(data_hash: int) -> str:
    matrix = _cached_cascade_matrix(data_hash)
    fig = plot_cascade_heatmap(matrix)
    return fig.to_json()


@lru_cache(maxsize=30)
def _cached_curve_json(trigger_region: str, data_hash: int) -> str:
    from ukraine_alerts.api.dependencies import get_cleaned_data as _gcd
    df = _gcd()
    curve = compute_secondary_strike_curve(df, trigger_region)
    fig = plot_secondary_strike_curve(curve, trigger_region)
    return fig.to_json()


@lru_cache(maxsize=1)
def _cached_regions(data_hash: int) -> list:
    matrix = _cached_cascade_matrix(data_hash)
    return sorted(list(matrix.index))


@router.get("/heatmap")
def get_cascade_heatmap(df: pd.DataFrame = Depends(get_cleaned_data)):
    json_str = _cached_heatmap_json(_df_hash(df))
    return Response(content=json_str, media_type="application/json")


@router.get("/secondary-curve")
def get_secondary_curve(trigger_region: str, df: pd.DataFrame = Depends(get_cleaned_data)):
    json_str = _cached_curve_json(trigger_region, _df_hash(df))
    return Response(content=json_str, media_type="application/json")


@router.get("/regions")
def get_cascade_regions(df: pd.DataFrame = Depends(get_cleaned_data)):
    regions = _cached_regions(_df_hash(df))
    return JSONResponse(content={"regions": regions})
