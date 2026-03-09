"""Layer CRUD and ingestion service."""
from __future__ import annotations

import json
import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.models import Layer


async def get_layers_for_session(
    session_id: str, db: AsyncSession
) -> list[dict[str, Any]]:
    result = await db.execute(
        select(Layer)
        .where(Layer.session_id == uuid.UUID(session_id))
        .order_by(Layer.created_at.desc())
    )
    layers = result.scalars().all()
    return [
        {
            "id": lyr.id,
            "name": lyr.name,
            "layer_type": lyr.layer_type,
            "metadata": lyr.metadata_ or {},
            "has_geodata": lyr.geodata_json is not None,
        }
        for lyr in layers
    ]
