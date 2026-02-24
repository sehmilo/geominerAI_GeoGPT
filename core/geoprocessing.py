"""
core/geoprocessing.py
Spatial geoprocessing utilities for GeoMinerAI.

Operations:
  buffer_geojson   – expand polygon/point features by N metres
  clip_points_to_polygon – keep only points inside a polygon
  compute_centroid  – centroid of a GeoJSON feature collection
  drawn_to_geojson  – normalise streamlit-folium draw output

Uses shapely when available; falls back to simple bbox / degree-math otherwise.
"""
from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── optional shapely ──────────────────────────────────────────────────────────
try:
    from shapely.geometry import Point, Polygon, MultiPolygon, shape, mapping
    from shapely.ops import unary_union
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False


# ── coordinate helpers ────────────────────────────────────────────────────────

def _deg_to_m(lat: float) -> Tuple[float, float]:
    """Return (m_per_lon_deg, m_per_lat_deg) at *lat*."""
    R        = 6_371_000.0
    lat_r    = math.radians(lat)
    m_per_lat = R * math.pi / 180.0
    m_per_lon = R * math.cos(lat_r) * math.pi / 180.0
    return m_per_lon, m_per_lat


def _avg_deg_per_m(lat: float) -> float:
    m_lon, m_lat = _deg_to_m(lat)
    return 1.0 / ((m_lon + m_lat) / 2.0)


# ── buffer ────────────────────────────────────────────────────────────────────

def buffer_geojson(geojson: Dict, distance_m: float) -> Dict:
    """
    Buffer every feature in *geojson* by *distance_m* metres.

    Returns a new GeoJSON FeatureCollection.
    Uses shapely if available, otherwise a simple degree approximation.
    """
    features = geojson.get("features", [])
    if not features:
        return {"type": "FeatureCollection", "features": []}

    if HAS_SHAPELY:
        return _buffer_shapely(features, distance_m)
    return _buffer_simple(features, distance_m)


def _buffer_shapely(features: list, distance_m: float) -> Dict:
    buffered = []
    for feat in features:
        geom = shape(feat["geometry"])
        lat  = geom.centroid.y
        deg  = _avg_deg_per_m(lat)
        buf  = geom.buffer(distance_m * deg)
        buffered.append({
            "type":       "Feature",
            "geometry":   mapping(buf),
            "properties": {**(feat.get("properties") or {}),
                           "_buffer_m": distance_m},
        })
    return {"type": "FeatureCollection", "features": buffered}


def _buffer_simple(features: list, distance_m: float) -> Dict:
    """Circle approximation for point features; bbox expansion for polygons."""
    buffered = []
    for feat in features:
        geom = feat.get("geometry", {})
        gtype = geom.get("type", "")

        if gtype == "Point":
            lon, lat   = geom["coordinates"]
            m_lon, m_lat = _deg_to_m(lat)
            d_lon = distance_m / m_lon
            d_lat = distance_m / m_lat
            angles = np.linspace(0, 2 * np.pi, 33)
            ring   = [[lon + d_lon * np.cos(a), lat + d_lat * np.sin(a)]
                      for a in angles]
            buffered.append({
                "type":     "Feature",
                "geometry": {"type": "Polygon", "coordinates": [ring]},
                "properties": {**(feat.get("properties") or {}),
                               "_buffer_m": distance_m},
            })
        else:
            # bbox expansion fallback
            coords = _all_coords(geom)
            if not coords:
                buffered.append(feat)
                continue
            lat0   = np.mean([c[1] for c in coords])
            m_lon, m_lat = _deg_to_m(lat0)
            d_lon = distance_m / m_lon
            d_lat = distance_m / m_lat
            min_lon = min(c[0] for c in coords) - d_lon
            max_lon = max(c[0] for c in coords) + d_lon
            min_lat = min(c[1] for c in coords) - d_lat
            max_lat = max(c[1] for c in coords) + d_lat
            ring = [[min_lon, min_lat], [max_lon, min_lat],
                    [max_lon, max_lat], [min_lon, max_lat],
                    [min_lon, min_lat]]
            buffered.append({
                "type":     "Feature",
                "geometry": {"type": "Polygon", "coordinates": [ring]},
                "properties": {**(feat.get("properties") or {}),
                               "_buffer_m": distance_m},
            })
    return {"type": "FeatureCollection", "features": buffered}


# ── clip ──────────────────────────────────────────────────────────────────────

def clip_points_to_polygon(
    df: pd.DataFrame,
    polygon_geojson: Dict,
    lat_col: str = "lat",
    lon_col: str = "lon",
) -> pd.DataFrame:
    """
    Return rows of *df* whose (lat, lon) fall inside any polygon in
    *polygon_geojson*.  Requires lat/lon columns.
    """
    features = polygon_geojson.get("features", [])
    if not features:
        return df

    if HAS_SHAPELY:
        polys = []
        for feat in features:
            try:
                polys.append(shape(feat["geometry"]))
            except Exception:
                pass
        if not polys:
            return df
        union = unary_union(polys)
        mask  = df.apply(
            lambda r: union.contains(Point(float(r[lon_col]), float(r[lat_col]))),
            axis=1,
        )
        return df[mask].copy()

    # Fallback: bounding-box clip using first polygon
    return _clip_bbox(df, features[0], lat_col, lon_col)


def _clip_bbox(df: pd.DataFrame, feature: Dict,
               lat_col: str, lon_col: str) -> pd.DataFrame:
    coords = _all_coords(feature.get("geometry", {}))
    if not coords:
        return df
    min_lon = min(c[0] for c in coords)
    max_lon = max(c[0] for c in coords)
    min_lat = min(c[1] for c in coords)
    max_lat = max(c[1] for c in coords)
    mask = (
        (df[lat_col] >= min_lat) & (df[lat_col] <= max_lat) &
        (df[lon_col] >= min_lon) & (df[lon_col] <= max_lon)
    )
    return df[mask].copy()


# ── centroid ──────────────────────────────────────────────────────────────────

def compute_centroid(geojson: Dict) -> Optional[Tuple[float, float]]:
    """Return (lat, lon) centroid of a GeoJSON FeatureCollection, or None."""
    coords = []
    for feat in geojson.get("features", []):
        coords.extend(_all_coords(feat.get("geometry", {})))
    if not coords:
        return None
    lats = [c[1] for c in coords]
    lons = [c[0] for c in coords]
    return float(np.mean(lats)), float(np.mean(lons))


# ── normalise streamlit-folium draw output ────────────────────────────────────

def drawn_to_geojson(draw_data: Any) -> Dict:
    """
    Accept whatever st_folium returns for 'all_drawings' and return a
    guaranteed GeoJSON FeatureCollection dict.
    """
    if draw_data is None:
        return {"type": "FeatureCollection", "features": []}

    if isinstance(draw_data, str):
        try:
            draw_data = json.loads(draw_data)
        except Exception:
            return {"type": "FeatureCollection", "features": []}

    if isinstance(draw_data, dict):
        if draw_data.get("type") == "FeatureCollection":
            return draw_data
        if draw_data.get("type") == "Feature":
            return {"type": "FeatureCollection", "features": [draw_data]}

    if isinstance(draw_data, list):
        return {"type": "FeatureCollection", "features": draw_data}

    return {"type": "FeatureCollection", "features": []}


# ── measure ───────────────────────────────────────────────────────────────────

def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres."""
    R    = 6_371_000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl   = math.radians(lon2 - lon1)
    a    = (math.sin(dphi / 2) ** 2
            + math.cos(phi1) * math.cos(phi2) * math.sin(dl / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(a))


def nearest_row(df: pd.DataFrame,
                click_lat: float, click_lon: float,
                lat_col: str = "lat",
                lon_col: str = "lon"):
    """Return (row, distance_m) for the point closest to the click."""
    dists = df.apply(
        lambda r: haversine_m(click_lat, click_lon,
                               float(r[lat_col]), float(r[lon_col])),
        axis=1,
    )
    i = int(dists.idxmin())
    return df.loc[i], float(dists.loc[i])


# ── internal ──────────────────────────────────────────────────────────────────

def _all_coords(geom: Dict) -> List[List[float]]:
    """Flatten all coordinate pairs from a GeoJSON geometry."""
    gtype = geom.get("type", "")
    raw   = geom.get("coordinates", [])
    if gtype == "Point":
        return [raw]
    if gtype in ("LineString", "MultiPoint"):
        return list(raw)
    if gtype == "Polygon":
        return [c for ring in raw for c in ring]
    if gtype == "MultiLineString":
        return [c for line in raw for c in line]
    if gtype == "MultiPolygon":
        return [c for poly in raw for ring in poly for c in ring]
    return []
