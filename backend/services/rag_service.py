"""pgvector-based RAG retrieval service (replaces FAISS)."""
from __future__ import annotations

import sys
import uuid
from pathlib import Path
from typing import Any

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

# Make core/ importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.db.models import Layer, TextChunk


def _get_embedder():
    """Lazily load SentenceTransformer to avoid loading it at import time."""
    from core.rag import EMBED
    return EMBED


async def embed_and_store(
    chunks: list[dict[str, Any]],
    layer_id: int,
    db: AsyncSession,
) -> int:
    """Encode text chunks with SentenceTransformer and store in pgvector."""
    if not chunks:
        return 0

    embedder = _get_embedder()
    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    for i, chunk in enumerate(chunks):
        tc = TextChunk(
            layer_id=layer_id,
            source=chunk.get("source", "unknown"),
            page=chunk.get("page", 1),
            text=chunk["text"],
            embedding=embeddings[i].tolist(),
        )
        db.add(tc)

    await db.flush()
    return len(chunks)


async def retrieve(
    question: str,
    session_id: str,
    db: AsyncSession,
    k: int = 5,
) -> list[dict[str, Any]]:
    """Query pgvector for top-k similar chunks across all session layers."""
    embedder = _get_embedder()
    q_emb = embedder.encode([question], convert_to_numpy=True, normalize_embeddings=True)
    q_vec = q_emb[0].tolist()

    # Get all layer IDs for this session
    result = await db.execute(
        select(Layer.id).where(Layer.session_id == uuid.UUID(session_id))
    )
    layer_ids = [r[0] for r in result.all()]

    if not layer_ids:
        return []

    # pgvector cosine similarity query
    query = text("""
        SELECT source, page, text,
               1 - (embedding <=> :q_vec::vector) AS similarity
        FROM text_chunks
        WHERE layer_id = ANY(:layer_ids)
          AND embedding IS NOT NULL
        ORDER BY embedding <=> :q_vec::vector
        LIMIT :k
    """)

    result = await db.execute(query, {
        "q_vec": str(q_vec),
        "layer_ids": layer_ids,
        "k": k,
    })
    rows = result.all()

    return [
        {"source": row.source, "page": row.page, "text": row.text}
        for row in rows
    ]
