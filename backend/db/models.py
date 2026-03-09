"""SQLAlchemy ORM models for GeominerAI."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from geoalchemy2 import Geometry
from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class Session(Base):
    __tablename__ = "sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    settings = Column(JSONB, default=dict)

    layers = relationship("Layer", back_populates="session", cascade="all, delete-orphan")
    outputs = relationship("Output", back_populates="session", cascade="all, delete-orphan")
    chat_messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")


class Layer(Base):
    __tablename__ = "layers"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(UUID(as_uuid=True), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    name = Column(String(512), nullable=False)
    layer_type = Column(String(50), nullable=False)
    metadata_ = Column("metadata", JSONB, default=dict)
    storage_path = Column(String(1024), nullable=True)
    dataframe_json = Column(Text, nullable=True)
    geodata_json = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    session = relationship("Session", back_populates="layers")
    chunks = relationship("TextChunk", back_populates="layer", cascade="all, delete-orphan")
    features = relationship("LayerFeature", back_populates="layer", cascade="all, delete-orphan")


class TextChunk(Base):
    __tablename__ = "text_chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    layer_id = Column(Integer, ForeignKey("layers.id", ondelete="CASCADE"), nullable=False)
    source = Column(String(512), nullable=False)
    page = Column(Integer, default=1)
    text = Column(Text, nullable=False)
    embedding = Column(Vector(384), nullable=True)

    layer = relationship("Layer", back_populates="chunks")


class Output(Base):
    __tablename__ = "outputs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(UUID(as_uuid=True), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    title = Column(String(512), nullable=False)
    output_type = Column(String(50), nullable=False)
    content_json = Column(Text, nullable=True)
    figure_path = Column(String(1024), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    session = relationship("Session", back_populates="outputs")


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(UUID(as_uuid=True), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    session = relationship("Session", back_populates="chat_messages")


class LayerFeature(Base):
    __tablename__ = "layer_features"

    id = Column(Integer, primary_key=True, autoincrement=True)
    layer_id = Column(Integer, ForeignKey("layers.id", ondelete="CASCADE"), nullable=False)
    geom = Column(Geometry(geometry_type="GEOMETRY", srid=4326), nullable=True)
    properties = Column(JSONB, default=dict)

    layer = relationship("Layer", back_populates="features")
