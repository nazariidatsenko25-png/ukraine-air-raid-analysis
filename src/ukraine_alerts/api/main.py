import json
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from plotly.utils import PlotlyJSONEncoder

from ukraine_alerts.api.routers import cascade, eda, models, threats


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup actions (e.g., loading data into memory if needed, but we will let routers handle it or cache it)
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
