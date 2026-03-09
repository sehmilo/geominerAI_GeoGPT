"""Chat endpoints with SSE streaming support."""
from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.deps import get_db, get_session_id
from backend.api.schemas.chat import ChatMessageCreate, ChatMessageRead
from backend.db.models import ChatMessage

router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.get("/")
async def list_messages(
    session_id: str = Depends(get_session_id),
    db: AsyncSession = Depends(get_db),
    limit: int = 50,
):
    result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.session_id == uuid.UUID(session_id))
        .order_by(ChatMessage.created_at.asc())
        .limit(limit)
    )
    messages = result.scalars().all()
    return [
        {
            "id": m.id,
            "role": m.role,
            "content": m.content,
            "created_at": m.created_at.isoformat() if m.created_at else None,
        }
        for m in messages
    ]


@router.post("/")
async def add_message(
    body: ChatMessageCreate,
    db: AsyncSession = Depends(get_db),
):
    msg = ChatMessage(
        session_id=uuid.UUID(body.session_id),
        role=body.role,
        content=body.content,
    )
    db.add(msg)
    await db.flush()
    return {
        "id": msg.id,
        "role": msg.role,
        "content": msg.content,
        "created_at": msg.created_at.isoformat() if msg.created_at else None,
    }


@router.get("/stream")
async def stream_tokens(task_id: str, request: Request):
    """SSE endpoint for streaming LLM tokens. Phase 3 implementation."""
    import asyncio

    redis = request.app.state.redis

    async def event_generator():
        channel = f"sse:{task_id}"
        pubsub = redis.pubsub()
        await pubsub.subscribe(channel)
        try:
            while True:
                if await request.is_disconnected():
                    break
                message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                if message and message["type"] == "message":
                    data = message["data"]
                    if isinstance(data, bytes):
                        data = data.decode()
                    if data == "[DONE]":
                        yield f"data: [DONE]\n\n"
                        break
                    yield f"data: {data}\n\n"
                else:
                    await asyncio.sleep(0.1)
        finally:
            await pubsub.unsubscribe(channel)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
