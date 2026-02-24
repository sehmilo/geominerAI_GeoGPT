"""
core/geo_ingest.py
Multi-format knowledge-base ingestion for GeoMinerAI.

Supported formats:
  PDF, Word (.docx), CSV, GeoJSON, KML, plain text,
  Images (PNG/JPG/TIFF/BMP), Raster (TIF/DEM/ASC), Shapefiles (info only)
"""
from __future__ import annotations

import io
import json
import re
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from pypdf import PdfReader

# ── optional imports handled gracefully ──────────────────────────────────────
try:
    from docx import Document as _DocxDoc
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False

try:
    from PIL import Image as _PILImage
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


# ── public API ───────────────────────────────────────────────────────────────

def ingest_file(uploaded_file) -> Dict[str, Any]:
    """
    Ingest any supported uploaded file.

    Returns a *layer dict*:
    {
        "name":      str,
        "type":      str,          # pdf | word | csv | geojson | kml | image
                                   # raster | vector | text | unknown
        "chunks":    List[Dict],   # [{source, page, text}, ...]
        "dataframe": pd.DataFrame | None,
        "geodata":   dict | None,  # parsed GeoJSON feature-collection
        "image":     PIL.Image | None,
        "metadata":  dict,
        "raw_bytes": bytes,
    }
    """
    raw: bytes = uploaded_file.read()
    name: str = uploaded_file.name
    ext: str = name.rsplit(".", 1)[-1].lower() if "." in name else ""

    layer: Dict[str, Any] = {
        "name":      name,
        "type":      "unknown",
        "chunks":    [],
        "dataframe": None,
        "geodata":   None,
        "image":     None,
        "metadata":  {"size_bytes": len(raw)},
        "raw_bytes": raw,
    }

    if ext == "pdf":
        layer["type"]   = "pdf"
        layer["chunks"] = _ingest_pdf(raw, name)

    elif ext in ("docx", "doc"):
        layer["type"]   = "word"
        layer["chunks"] = _ingest_word(raw, name)

    elif ext == "csv":
        layer["type"]      = "csv"
        layer["dataframe"] = _safe_read_csv(raw)
        layer["chunks"]    = _csv_to_chunks(layer["dataframe"], name)

    elif ext in ("geojson", "json"):
        layer["type"]    = "geojson"
        geodata          = json.loads(raw.decode("utf-8", errors="replace"))
        layer["geodata"] = geodata
        layer["chunks"]  = _geojson_to_chunks(geodata, name)

    elif ext == "kml":
        layer["type"]   = "kml"
        layer["chunks"] = _kml_to_chunks(raw, name)

    elif ext in ("tif", "tiff", "img", "asc", "dem", "grd"):
        layer["type"] = "raster"
        layer["metadata"].update(_raster_meta(raw, name))

    elif ext in ("png", "jpg", "jpeg", "bmp", "gif", "webp", "tiff"):
        layer["type"] = "image"
        if HAS_PIL:
            img = _PILImage.open(io.BytesIO(raw))
            layer["image"] = img
            layer["metadata"]["mode"] = img.mode
            layer["metadata"]["size_px"] = img.size
        layer["chunks"] = [{"source": name, "page": 1,
                             "text": f"Image file: {name} (visual content — "
                                     f"{layer['metadata'].get('size_px', 'unknown')} px)"}]

    elif ext in ("shp", "gpkg", "gdb"):
        layer["type"]   = "vector"
        layer["chunks"] = [{"source": name, "page": 1,
                             "text": f"Vector geodata: {name}. "
                                     f"Convert to GeoJSON for full spatial support."}]

    elif ext in ("txt", "md", "log"):
        layer["type"]   = "text"
        layer["chunks"] = _text_to_chunks(raw.decode("utf-8", errors="replace"), name)

    else:
        # Attempt plain-text parse as last resort
        try:
            text = raw.decode("utf-8", errors="replace")
            layer["type"]   = "text"
            layer["chunks"] = _text_to_chunks(text, name)
        except Exception:
            layer["chunks"] = [{"source": name, "page": 1,
                                 "text": f"Unknown file: {name} ({len(raw)} bytes)"}]

    return layer


def build_index_from_layers(layers: List[Dict]):
    """
    Build (or rebuild) a FAISS index over all text chunks in *layers*.

    Returns (index, all_chunks) — same shape as build_index_from_pdfs().
    """
    import faiss
    from core.rag import EMBED

    all_chunks: List[Dict] = []
    for lyr in layers:
        all_chunks.extend(lyr.get("chunks", []))

    if not all_chunks:
        return None, []

    texts = [c["text"] for c in all_chunks]
    emb   = EMBED.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    dim   = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb.astype(np.float32))
    return index, all_chunks


def get_csv_layers(layers: List[Dict]) -> List[Dict]:
    """Return all CSV-type layers that have a dataframe."""
    return [l for l in layers if l["type"] == "csv" and l["dataframe"] is not None]


def get_geojson_layers(layers: List[Dict]) -> List[Dict]:
    """Return all layers that carry a GeoJSON feature-collection."""
    return [l for l in layers if l.get("geodata") is not None]


# ── internal helpers ─────────────────────────────────────────────────────────

def _chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> List[str]:
    chunks, start, n = [], 0, len(text)
    while start < n:
        end   = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
        if start < 0:
            start = 0
        if end == n:
            break
    return chunks


def _ingest_pdf(raw: bytes, name: str) -> List[Dict]:
    reader = PdfReader(io.BytesIO(raw))
    chunks = []
    for i, page in enumerate(reader.pages):
        text = " ".join((page.extract_text() or "").split())
        if text:
            for ch in _chunk_text(text):
                chunks.append({"source": name, "page": i + 1, "text": ch})
    return chunks


def _ingest_word(raw: bytes, name: str) -> List[Dict]:
    if not HAS_DOCX:
        return [{"source": name, "page": 1,
                 "text": f"Word file: {name} — install python-docx for full text extraction."}]
    doc       = _DocxDoc(io.BytesIO(raw))
    full_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    return [{"source": name, "page": i + 1, "text": ch}
            for i, ch in enumerate(_chunk_text(full_text))]


def _safe_read_csv(raw: bytes) -> pd.DataFrame:
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc)
        except Exception:
            continue
    return pd.DataFrame()


def _csv_to_chunks(df: pd.DataFrame, name: str) -> List[Dict]:
    summary = (f"CSV: {name}. Columns: {list(df.columns)}. "
               f"Rows: {len(df)}. "
               f"Lat/lon available: {'lat' in df.columns and 'lon' in df.columns}.")
    try:
        stats = df.describe(include="all").to_string()[:1200]
    except Exception:
        stats = ""
    chunks = [{"source": name, "page": 1, "text": summary}]
    if stats:
        chunks.append({"source": name, "page": 2, "text": f"Statistics:\n{stats}"})
    return chunks


def _geojson_to_chunks(data: Dict, name: str) -> List[Dict]:
    n    = len(data.get("features", [])) if "features" in data else 0
    gtypes = list({f.get("geometry", {}).get("type", "?")
                   for f in data.get("features", [])})
    text = f"GeoJSON: {name}. Features: {n}. Geometry types: {gtypes}."
    if n > 0:
        props = data["features"][0].get("properties", {})
        text += f" Sample properties: {list(props.keys())[:10]}"
    return [{"source": name, "page": 1, "text": text}]


def _kml_to_chunks(raw: bytes, name: str) -> List[Dict]:
    text  = raw.decode("utf-8", errors="replace")
    clean = " ".join(re.sub(r"<[^>]+>", " ", text).split())
    return [{"source": name, "page": 1,
             "text": f"KML geodata: {name}. Content preview: {clean[:800]}"}]


def _raster_meta(raw: bytes, name: str) -> Dict:
    if not HAS_RASTERIO:
        return {"note": f"Raster: {name}. Install rasterio for metadata."}
    try:
        with rasterio.open(io.BytesIO(raw)) as src:
            return {
                "crs":       str(src.crs),
                "bounds":    list(src.bounds),
                "width":     src.width,
                "height":    src.height,
                "bands":     src.count,
                "dtype":     str(src.dtypes[0]),
                "transform": list(src.transform),
            }
    except Exception as e:
        return {"note": f"Raster: {name}. Could not parse: {e}"}


def _text_to_chunks(text: str, name: str) -> List[Dict]:
    return [{"source": name, "page": i + 1, "text": ch}
            for i, ch in enumerate(_chunk_text(text))]
