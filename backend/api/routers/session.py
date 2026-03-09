"""Session management endpoints."""
from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.deps import get_db
from backend.db.models import Session

router = APIRouter(prefix="/api/session", tags=["session"])


@router.post("/")
async def create_session(db: AsyncSession = Depends(get_db)):
    session = Session(id=uuid.uuid4(), settings={})
    db.add(session)
    await db.flush()
    return {"session_id": str(session.id)}


@router.get("/{session_id}")
async def get_session(session_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Session).where(Session.id == uuid.UUID(session_id)))
    session = result.scalar_one_or_none()
    if not session:
        return {"error": "Session not found"}, 404
    return {
        "session_id": str(session.id),
        "created_at": session.created_at.isoformat() if session.created_at else None,
        "settings": session.settings or {},
    }
