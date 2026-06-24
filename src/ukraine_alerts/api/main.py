import json
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from plotly.utils import PlotlyJSONEncoder

from ukraine_alerts.api.routers import cascade, eda, models, threats


from ukraine_alerts.api.dependencies import get_cleaned_data
import logging

logger = logging.getLogger("uvicorn.error")

@asynccontextmanager
async def lifespan(app: FastAPI):
    import threading
    # Preload the dataset so the first request is fast
    logger.info("Preloading cleaned dataset into memory...")
    get_cleaned_data()
    logger.info("Dataset loaded successfully.")

    # Warm model caches in background — first real user request will hit cache
    def _warm_caches():
        try:
            from ukraine_alerts.models.discretization import list_regions_with_data
            from ukraine_alerts.api.routers.models import _cached_regimes_json, _cached_forecast_json, _df_hash
            from ukraine_alerts.api.routers.cascade import _cached_heatmap_json, _df_hash as _c_hash
            df = get_cleaned_data()
            h = _df_hash(df)
            regions = list_regions_with_data(df, min_days=30)
            logger.info("Warming cascade heatmap cache...")
            _cached_heatmap_json(_c_hash(df))
            logger.info("Cascade heatmap cached.")
            for r in regions[:5]:  # warm top 5 regions
                logger.info("Warming model cache for %s...", r)
                _cached_regimes_json(r, h)
                _cached_forecast_json(r, h)
            logger.info("Model cache warm-up complete.")
        except Exception as e:
            logger.warning("Cache warm-up failed (non-fatal): %s", e)

    threading.Thread(target=_warm_caches, daemon=True).start()
    yield
    # Shutdown actions

app = FastAPI(
    title="Ukraine Alerts API",
    description="Backend API for Ukraine Air Raid Analysis",
    version="1.0.0",
    lifespan=lifespan,
)

import os

FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL, "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(eda.router, prefix="/api/v1/eda", tags=["EDA"])
app.include_router(cascade.router, prefix="/api/v1/cascade", tags=["Cascade"])
app.include_router(models.router, prefix="/api/v1/models", tags=["Models"])
app.include_router(threats.router, prefix="/api/v1/threats", tags=["Threats"])


@app.get("/health")
async def health_check():
    return {"status": "ok"}
