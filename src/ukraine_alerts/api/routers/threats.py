from fastapi import APIRouter, Depends, Response
import pandas as pd

from ukraine_alerts.api.dependencies import get_cleaned_data
from ukraine_alerts.models.threat_clustering import group_attack_waves, fit_threat_gmm
from ukraine_alerts.charts.threat_charts import plot_threat_scatter, plot_threat_timeline

router = APIRouter()

@router.get("/scatter")
def get_threat_scatter(df: pd.DataFrame = Depends(get_cleaned_data)):
    waves = group_attack_waves(df)
    waves, _, _ = fit_threat_gmm(waves)
    fig = plot_threat_scatter(waves)
    return Response(content=fig.to_json(), media_type="application/json")

@router.get("/timeline")
def get_threat_timeline(df: pd.DataFrame = Depends(get_cleaned_data)):
    waves = group_attack_waves(df)
    waves, _, _ = fit_threat_gmm(waves)
    fig = plot_threat_timeline(waves)
    return Response(content=fig.to_json(), media_type="application/json")
