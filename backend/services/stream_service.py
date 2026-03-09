"""SSE streaming service for LLM token-by-token output (Phase 3)."""
from __future__ import annotations

import json
from typing import AsyncGenerator

import redis.asyncio as aioredis


async def publish_token(task_id: str, token: str, redis: aioredis.Redis):
    """Publish a single LLM token to the SSE channel."""
    await redis.publish(f"sse:{task_id}", token)


async def publish_done(task_id: str, redis: aioredis.Redis):
    """Signal that LLM generation is complete."""
    await redis.publish(f"sse:{task_id}", "[DONE]")


async def subscribe_tokens(
    task_id: str, redis: aioredis.Redis
) -> AsyncGenerator[str, None]:
    """Subscribe to SSE channel and yield tokens."""
    channel = f"sse:{task_id}"
    pubsub = redis.pubsub()
    await pubsub.subscribe(channel)
    try:
        async for message in pubsub.listen():
            if message["type"] == "message":
                data = message["data"]
                if isinstance(data, bytes):
                    data = data.decode()
                if data == "[DONE]":
                    break
                yield data
    finally:
        await pubsub.unsubscribe(channel)
