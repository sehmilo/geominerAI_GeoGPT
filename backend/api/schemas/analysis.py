"""Pydantic schemas for analysis endpoints."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class DispatchRequest(BaseModel):
    prompt: str
    session_id: str


class DispatchResponse(BaseModel):
    intent: str
    metadata: dict[str, Any] = {}
    outputs: list[AnalysisOutput] = []


class QARequest(BaseModel):
    prompt: str
    session_id: str


class QAResponse(BaseModel):
    answer: str
    evidence: list[dict[str, Any]] = []


class HotspotRequest(BaseModel):
    prompt: str
    session_id: str
    method: str = "Gi*"
    variable: str = "SnO2"


class CrossSectionRequest(BaseModel):
    prompt: str
    session_id: str
    section_label: str = "A — A'"
    section_length_m: float = 1000.0
    drawn_features: dict[str, Any] | None = None


class BufferRequest(BaseModel):
    session_id: str
    distance_m: float = 500.0
    drawn_features: dict[str, Any]


class ClipRequest(BaseModel):
    session_id: str
    drawn_features: dict[str, Any]


class ProspectivityRequest(BaseModel):
    prompt: str
    session_id: str


class DepthProfileRequest(BaseModel):
    prompt: str
    session_id: str
    variable: str = "SnO2"


class AnalysisOutput(BaseModel):
    title: str
    output_type: str
    content: Any = None
    figure_url: str | None = None
    timestamp: str = ""


# Forward reference resolution
DispatchResponse.model_rebuild()
