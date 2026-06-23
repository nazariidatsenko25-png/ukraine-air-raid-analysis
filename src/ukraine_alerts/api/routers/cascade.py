from fastapi import APIRouter, Depends, Response
from fastapi.responses import JSONResponse
import pandas as pd

from ukraine_alerts.api.dependencies import get_cleaned_data
from ukraine_alerts.eda.cascade import compute_cascade_matrix, compute_secondary_strike_curve
from ukraine_alerts.charts.cascade_charts import plot_cascade_heatmap, plot_secondary_strike_curve

router = APIRouter()

@router.get("/heatmap")
def get_cascade_heatmap(df: pd.DataFrame = Depends(get_cleaned_data)):
    matrix, _ = compute_cascade_matrix(df, window_hours=3)
    fig = plot_cascade_heatmap(matrix)
    return Response(content=fig.to_json(), media_type="application/json")

@router.get("/secondary-curve")
def get_secondary_curve(trigger_region: str, df: pd.DataFrame = Depends(get_cleaned_data)):
    matrix, _ = compute_cascade_matrix(df, window_hours=3)
    # The compute_secondary_strike_curve expects the raw df, not the matrix?
    # Wait, in the Streamlit app: curve = compute_secondary_strike_curve(df, selected_trigger)
    curve = compute_secondary_strike_curve(df, trigger_region)
    fig = plot_secondary_strike_curve(curve, trigger_region)
    return Response(content=fig.to_json(), media_type="application/json")

@router.get("/regions")
def get_cascade_regions(df: pd.DataFrame = Depends(get_cleaned_data)):
    matrix, _ = compute_cascade_matrix(df, window_hours=3)
    regions = sorted(list(matrix.index))
    return JSONResponse(content={"regions": regions})
