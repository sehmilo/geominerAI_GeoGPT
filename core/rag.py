from __future__ import annotations

import io
from typing import List, Dict, Any, Tuple

import numpy as np
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

EMBED = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def read_pdf_pages(file_bytes: bytes) -> List[Dict[str, Any]]:
    reader = PdfReader(io.BytesIO(file_bytes))
    out = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = " ".join(text.split())
        if text:
            out.append({"page": i + 1, "text": text})
    return out


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> List[str]:
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be larger than overlap")

    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap
        if start < 0:
            start = 0
        if end == n:
            break

    return chunks


def build_index_from_pdfs(uploaded_files) -> Tuple[faiss.IndexFlatIP, List[Dict[str, Any]]]:
    chunks: List[Dict[str, Any]] = []

    for f in uploaded_files:
        file_bytes = f.read()
        pages = read_pdf_pages(file_bytes)

        for p in pages:
            for ch in chunk_text(p["text"]):
                chunks.append(
                    {"source": f.name, "page": p["page"], "text": ch}
                )

    texts = [c["text"] for c in chunks]
    emb = EMBED.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb.astype(np.float32))

    return index, chunks


def retrieve(question: str, index: faiss.IndexFlatIP, chunks: List[Dict[str, Any]], k: int = 5) -> List[Dict[str, Any]]:
    q_emb = EMBED.encode([question], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    _, ids = index.search(q_emb, k)

    results = []
    for idx in ids[0]:
        if idx == -1:
            continue
        results.append(chunks[int(idx)])

    return results
