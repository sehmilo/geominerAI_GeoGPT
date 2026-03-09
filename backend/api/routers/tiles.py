"""MVT tile serving endpoint (Phase 4)."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.deps import get_db

router = APIRouter(prefix="/api/tiles", tags=["tiles"])


@router.get("/{layer_id}/{z}/{x}/{y}.mvt")
async def get_tile(
    layer_id: int,
    z: int,
    x: int,
    y: int,
    db: AsyncSession = Depends(get_db),
):
    """Serve Mapbox Vector Tiles from PostGIS layer_features table."""
    query = text("""
        SELECT ST_AsMVT(q, 'layer', 4096, 'geom')
        FROM (
            SELECT id, properties,
                   ST_AsMVTGeom(
                       geom,
                       ST_TileEnvelope(:z, :x, :y),
                       4096, 256, true
                   ) AS geom
            FROM layer_features
            WHERE layer_id = :layer_id
              AND ST_Intersects(geom, ST_TileEnvelope(:z, :x, :y))
        ) q
    """)

    result = await db.execute(query, {"layer_id": layer_id, "z": z, "x": x, "y": y})
    tile_data = result.scalar()

    if not tile_data:
        return Response(content=b"", media_type="application/vnd.mapbox-vector-tile")

    return Response(
        content=bytes(tile_data),
        media_type="application/vnd.mapbox-vector-tile",
        headers={"Cache-Control": "public, max-age=3600"},
    )
