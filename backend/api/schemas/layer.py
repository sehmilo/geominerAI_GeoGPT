"""Pydantic schemas for layer endpoints."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel


class LayerCreate(BaseModel):
    name: str
    layer_type: str
    metadata: dict[str, Any] = {}


class LayerRead(BaseModel):
    id: int
    name: str
    layer_type: str
    metadata: dict[str, Any] = {}
    created_at: datetime | None = None
    row_count: int | None = None
    has_geodata: bool = False

    model_config = {"from_attributes": True}
