"""Pydantic schemas for chat endpoints."""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class ChatMessageCreate(BaseModel):
    role: str
    content: str
    session_id: str


class ChatMessageRead(BaseModel):
    id: int
    role: str
    content: str
    created_at: datetime | None = None

    model_config = {"from_attributes": True}
