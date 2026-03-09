"""FastAPI dependency injection."""
from __future__ import annotations

import uuid
from typing import AsyncGenerator

import redis.asyncio as aioredis
from fastapi import Header, Request
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import settings
from backend.db.engine import async_session


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with async_session() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def get_redis(request: Request) -> aioredis.Redis:
    return request.app.state.redis


async def get_session_id(
    x_session_id: str | None = Header(default=None),
) -> str:
    if x_session_id:
        return x_session_id
    return str(uuid.uuid4())
