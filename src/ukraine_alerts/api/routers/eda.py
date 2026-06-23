from fastapi import APIRouter, Depends, Response
from fastapi.responses import JSONResponse
import pandas as pd

from ukraine_alerts.api.dependencies import get_cleaned_data
from ukraine_alerts.charts.eda_charts import (
    plot_daily_alert_counts,
    plot_hourly_heatmap,
    plot_region_alert_ranking,
    plot_region_duration_comparison,
    plot_region_treemap,
)

router = APIRouter()

@router.get("/daily-volume")
def get_daily_volume(df: pd.DataFrame = Depends(get_cleaned_data)):
    fig = plot_daily_alert_counts(df)
    return Response(content=fig.to_json(), media_type="application/json")

@router.get("/heatmap")
def get_heatmap(df: pd.DataFrame = Depends(get_cleaned_data)):
    fig = plot_hourly_heatmap(df)
    return Response(content=fig.to_json(), media_type="application/json")

@router.get("/regional-ranking")
def get_regional_ranking(df: pd.DataFrame = Depends(get_cleaned_data)):
    fig = plot_region_alert_ranking(df)
    return Response(content=fig.to_json(), media_type="application/json")

@router.get("/regional-duration")
def get_regional_duration(df: pd.DataFrame = Depends(get_cleaned_data)):
    fig = plot_region_duration_comparison(df)
    return Response(content=fig.to_json(), media_type="application/json")

@router.get("/regional-treemap")
def get_regional_treemap(df: pd.DataFrame = Depends(get_cleaned_data)):
    fig = plot_region_treemap(df)
    return Response(content=fig.to_json(), media_type="application/json")
