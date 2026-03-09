"""Layer management endpoints."""
from __future__ import annotations

import json
import sys
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

# Make core/ importable
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from core.geo_ingest import ingest_file

from backend.api.deps import get_db, get_session_id
from backend.api.schemas.layer import LayerRead
from backend.db.models import Layer, Session, TextChunk
from backend.services.rag_service import embed_and_store

router = APIRouter(prefix="/api/layers", tags=["layers"])


class _UploadFileAdapter:
    """Adapts FastAPI UploadFile to the interface expected by core.geo_ingest.ingest_file."""

    def __init__(self, name: str, content: bytes):
        self.name = name
        self._content = content

    def read(self) -> bytes:
        return self._content


@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    session_id: str = Depends(get_session_id),
    db: AsyncSession = Depends(get_db),
):
    # Ensure session exists
    result = await db.execute(select(Session).where(Session.id == uuid.UUID(session_id)))
    session = result.scalar_one_or_none()
    if not session:
        session = Session(id=uuid.UUID(session_id), settings={})
        db.add(session)
        await db.flush()

    content = await file.read()
    adapter = _UploadFileAdapter(file.filename or "unknown", content)

    # Ingest via existing core module
    layer_dict = ingest_file(adapter)

    # Store layer in DB
    db_layer = Layer(
        session_id=uuid.UUID(session_id),
        name=layer_dict["name"],
        layer_type=layer_dict["type"],
        metadata_=layer_dict.get("metadata", {}),
    )

    # Store dataframe as JSON
    df = layer_dict.get("dataframe")
    if df is not None:
        db_layer.dataframe_json = df.to_json(orient="split", default_handler=str)

    # Store geodata as JSON
    geodata = layer_dict.get("geodata")
    if geodata:
        db_layer.geodata_json = json.dumps(geodata)

    db.add(db_layer)
    await db.flush()

    # Embed text chunks into pgvector
    chunks = layer_dict.get("chunks", [])
    if chunks:
        await embed_and_store(chunks, db_layer.id, db)

    row_count = len(df) if df is not None else None
    return {
        "id": db_layer.id,
        "name": db_layer.name,
        "layer_type": db_layer.layer_type,
        "chunks_stored": len(chunks),
        "row_count": row_count,
        "has_geodata": geodata is not None,
    }


@router.get("/")
async def list_layers(
    session_id: str = Depends(get_session_id),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Layer)
        .where(Layer.session_id == uuid.UUID(session_id))
        .order_by(Layer.created_at.desc())
    )
    layers = result.scalars().all()
    out = []
    for lyr in layers:
        row_count = None
        if lyr.dataframe_json:
            try:
                import pandas as pd
                df = pd.read_json(BytesIO(lyr.dataframe_json.encode()), orient="split")
                row_count = len(df)
            except Exception:
                pass
        out.append({
            "id": lyr.id,
            "name": lyr.name,
            "layer_type": lyr.layer_type,
            "metadata": lyr.metadata_ or {},
            "created_at": lyr.created_at.isoformat() if lyr.created_at else None,
            "row_count": row_count,
            "has_geodata": lyr.geodata_json is not None,
        })
    return out


@router.get("/{layer_id}")
async def get_layer(layer_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Layer).where(Layer.id == layer_id))
    lyr = result.scalar_one_or_none()
    if not lyr:
        raise HTTPException(status_code=404, detail="Layer not found")
    return {
        "id": lyr.id,
        "name": lyr.name,
        "layer_type": lyr.layer_type,
        "metadata": lyr.metadata_ or {},
        "created_at": lyr.created_at.isoformat() if lyr.created_at else None,
        "has_geodata": lyr.geodata_json is not None,
    }


@router.delete("/{layer_id}")
async def delete_layer(layer_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Layer).where(Layer.id == layer_id))
    lyr = result.scalar_one_or_none()
    if not lyr:
        raise HTTPException(status_code=404, detail="Layer not found")
    await db.delete(lyr)
    return {"deleted": True, "id": layer_id}


@router.get("/{layer_id}/data")
async def get_layer_data(layer_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Layer).where(Layer.id == layer_id))
    lyr = result.scalar_one_or_none()
    if not lyr:
        raise HTTPException(status_code=404, detail="Layer not found")
    if not lyr.dataframe_json:
        raise HTTPException(status_code=404, detail="No dataframe data for this layer")
    import pandas as pd
    df = pd.read_json(BytesIO(lyr.dataframe_json.encode()), orient="split")
    return {"columns": list(df.columns), "data": df.to_dict(orient="records")}


@router.get("/{layer_id}/geojson")
async def get_layer_geojson(layer_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Layer).where(Layer.id == layer_id))
    lyr = result.scalar_one_or_none()
    if not lyr:
        raise HTTPException(status_code=404, detail="Layer not found")
    if not lyr.geodata_json:
        raise HTTPException(status_code=404, detail="No geodata for this layer")
    return json.loads(lyr.geodata_json)
