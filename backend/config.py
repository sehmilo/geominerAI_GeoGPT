"""Application configuration via environment variables."""
from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Database
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/geominerai"
    DATABASE_URL_SYNC: str = "postgresql://postgres:postgres@localhost:5432/geominerai"

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"

    # HuggingFace
    HF_TOKEN: str = ""
    HF_MODEL: str = "HuggingFaceH4/zephyr-7b-beta"
    HF_PROVIDER: str = "auto"

    # CORS
    CORS_ORIGINS: list[str] = ["http://localhost:5173", "http://localhost:3000"]

    # Auth
    SECRET_KEY: str = "change-me-in-production"

    # Embedding
    EMBEDDING_DIM: int = 384

    # Static files
    FIGURES_DIR: str = "static/figures"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
