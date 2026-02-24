"""
core/intent.py
Lightweight NLP intent detection for GeoMinerAI command routing.

Returns a (intent_name, metadata_dict) tuple so the caller can dispatch
to the right analysis handler without touching any LLM.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Tuple

# ── intent keyword tables ─────────────────────────────────────────────────────

_CROSS_SECTION = [
    "cross section", "cross-section", "crosssection", "section line",
    "geological section", "draw section", "structural section",
    "subsurface section", "a to a", "a to a'", "a–a", "a — a",
]

_HOTSPOT = [
    "hotspot", "hot spot", "spatial cluster", "spatial analysis",
    "gi*", "getis", "moran", "anomal", "geochemical pattern",
    "anomalous zone", "clustering", "heat map", "heatmap",
]

_BUFFER = [
    "buffer", "zone around", "distance from", "expand polygon",
    "offset polygon", "dilate",
]

_CLIP = [
    "clip", "cut to", "extract within", "intersect", "mask by",
    "points inside", "data inside",
]

_PROSPECTIVITY = [
    "prospectivity", "mineralization potential", "targeting",
    "sn-ree", "snree", "sn lab", "xrf", "tin potential",
    "rare earth", "tantalum", "niobium", "fracproxy",
    "greisen", "pegmatite", "granite system",
]

_MAP_DISPLAY = [
    "show on map", "add to map", "display layer", "plot layer",
    "visualize layer", "overlay",
]

_DEPTH_PROFILE = [
    "depth profile", "strip log", "downhole", "drill profile",
    "within hole", "interval profile",
]


# ── public API ────────────────────────────────────────────────────────────────

def detect_intent(text: str) -> Tuple[str, Dict[str, Any]]:
    """
    Classify *text* into one intent and return supporting metadata.

    Intents (in priority order):
      cross_section | hotspot | buffer | clip | prospectivity |
      depth_profile | map_display | qa
    """
    lower = text.lower()

    # 1. Cross-section (highest priority – flagship feature)
    if _any(lower, _CROSS_SECTION):
        return "cross_section", {
            "raw":              text,
            "section_label":    _extract_section_label(text),
            "section_length_m": _extract_distance_m(lower, default=1000.0),
        }

    # 2. Hotspot / spatial analysis
    if _any(lower, _HOTSPOT):
        method = "Gi*"
        if "moran" in lower:
            method = "Moran"
        elif "grid" in lower:
            method = "Grid"
        return "hotspot", {
            "raw":      text,
            "method":   method,
            "variable": _extract_geochem_var(lower),
        }

    # 3. Buffer
    if _any(lower, _BUFFER):
        dist = _extract_distance_m(lower, default=500.0)
        return "buffer", {
            "raw":        text,
            "distance_m": dist,
        }

    # 4. Clip
    if _any(lower, _CLIP):
        return "clip", {"raw": text}

    # 5. Prospectivity / Sn-REE lab
    if _any(lower, _PROSPECTIVITY):
        return "prospectivity", {"raw": text}

    # 6. Depth / drill profile
    if _any(lower, _DEPTH_PROFILE):
        return "depth_profile", {
            "raw":      text,
            "variable": _extract_geochem_var(lower),
        }

    # 7. Map display
    if _any(lower, _MAP_DISPLAY):
        return "map_display", {"raw": text}

    # 8. Default: RAG question-answering
    return "qa", {"raw": text}


# ── helpers ───────────────────────────────────────────────────────────────────

def _any(text: str, keywords: list) -> bool:
    return any(kw in text for kw in keywords)


def _extract_distance_m(lower: str, default: float = 500.0) -> float:
    """Extract a distance in metres from text (handles km, m suffixes)."""
    match = re.search(r"(\d+(?:\.\d+)?)\s*(km|kilometer|kilometre|m\b|meter|metre)", lower)
    if not match:
        return default
    val  = float(match.group(1))
    unit = match.group(2)
    if unit.startswith("k"):
        val *= 1000.0
    return val


def _extract_geochem_var(lower: str) -> str:
    """Identify which geochemical variable the user is referring to."""
    for candidate in ("sno2", "ta2o5", "nb2o5", "zro2", "fracproxy", "rb2o", "cs2o"):
        if candidate in lower:
            return candidate.upper().replace("SNO2", "SnO2").replace("TA2O5", "Ta2O5") \
                            .replace("NB2O5", "Nb2O5").replace("ZRO2", "ZrO2") \
                            .replace("FRACPROXY", "FracProxy").replace("RB2O", "Rb2O") \
                            .replace("CS2O", "Cs2O")
    for word in ("tin", "sn "):
        if word in lower:
            return "SnO2"
    for word in ("tantalum", "ta "):
        if word in lower:
            return "Ta2O5"
    for word in ("zirconium", "zr "):
        if word in lower:
            return "ZrO2"
    return "SnO2"  # default


def _extract_section_label(text: str) -> str:
    """Try to find a section-line label like 'A to A'' or 'B–B'' in the text."""
    # Pattern: single letter, optional prime/apostrophe
    match = re.search(
        r"\b([A-Z])\s*(?:to|–|—|-)\s*([A-Z][\'′]?)\b", text, re.IGNORECASE
    )
    if match:
        a, b = match.group(1).upper(), match.group(2).upper()
        return f"{a} — {b}"
    return "A — A′"
