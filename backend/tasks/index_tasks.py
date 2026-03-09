"""Celery tasks for embedding chunks (Phase 3: background indexing)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.tasks.celery_app import celery_app


@celery_app.task(name="index.embed_chunks")
def task_embed_chunks(chunks: list[dict], layer_id: int) -> dict:
    """Embed text chunks and store in pgvector via synchronous DB."""
    from core.rag import EMBED
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from backend.config import settings
    from backend.db.models import TextChunk

    if not chunks:
        return {"stored": 0}

    texts = [c["text"] for c in chunks]
    embeddings = EMBED.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    engine = create_engine(settings.DATABASE_URL_SYNC)
    Session = sessionmaker(bind=engine)

    with Session() as db:
        for i, chunk in enumerate(chunks):
            tc = TextChunk(
                layer_id=layer_id,
                source=chunk.get("source", "unknown"),
                page=chunk.get("page", 1),
                text=chunk["text"],
                embedding=embeddings[i].tolist(),
            )
            db.add(tc)
        db.commit()

    return {"stored": len(chunks)}
