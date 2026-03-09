"""GeominerAI FastAPI application."""
from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Make core/ importable from parent directory
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import redis.asyncio as aioredis
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    app.state.redis = aioredis.from_url(settings.REDIS_URL, decode_responses=True)

    # Create static/figures directory
    figures_dir = Path(__file__).parent / "static" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    yield

    # Shutdown
    await app.state.redis.close()


app = FastAPI(
    title="GeominerAI API",
    description="Geological analysis platform — FastAPI backend",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files for figure PNGs
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Routers
from backend.api.routers.health import router as health_router
from backend.api.routers.session import router as session_router
from backend.api.routers.layers import router as layers_router
from backend.api.routers.chat import router as chat_router
from backend.api.routers.analysis import router as analysis_router
from backend.api.routers.tiles import router as tiles_router

app.include_router(health_router)
app.include_router(session_router)
app.include_router(layers_router)
app.include_router(chat_router)
app.include_router(analysis_router)
app.include_router(tiles_router)


@app.get("/")
async def root():
    return {"service": "geominerai-api", "status": "running"}
