"""Analysis endpoints — mirrors all handlers from app.py."""
from __future__ import annotations

import io
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

# Make core/ importable
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from core.chemistry import ensure_oxides
from core.cross_section import (
    build_cross_section_prompt,
    build_layer_context,
    generate_cross_section_figure,
    parse_faults_from_text,
    parse_units_from_text,
)
from core.geoprocessing import buffer_geojson, clip_points_to_polygon
from core.intent import detect_intent
from core.llm_hf import answer_with_hf, generate_from_prompt
from core.snree_hotspots import hotspot_analysis
from core.snree_lab import (
    add_depth_interval_fields,
    build_snree_prompt,
    quick_prospectivity,
    summarize_xrf,
)
from core.snree_ml import build_features, train_baselines, within_hole_prediction

from backend.api.deps import get_db, get_session_id
from backend.api.schemas.analysis import (
    AnalysisOutput,
    BufferRequest,
    ClipRequest,
    CrossSectionRequest,
    DepthProfileRequest,
    DispatchRequest,
    DispatchResponse,
    HotspotRequest,
    ProspectivityRequest,
    QARequest,
    QAResponse,
)
from backend.config import settings
from backend.db.models import ChatMessage, Layer, Output
from backend.services.rag_service import retrieve

router = APIRouter(prefix="/api/analysis", tags=["analysis"])

FIGURES_DIR = Path(__file__).resolve().parents[2] / "static" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def _save_figure(fig, title: str) -> str:
    """Save matplotlib figure as PNG and return relative URL path."""
    import matplotlib
    matplotlib.use("Agg")

    filename = f"{title.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    filepath = FIGURES_DIR / filename
    fig.savefig(filepath, format="png", dpi=150, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return f"/static/figures/{filename}"


async def _get_session_layers(session_id: str, db: AsyncSession) -> list[dict]:
    """Load all layers for a session, hydrating dataframes and geodata."""
    result = await db.execute(
        select(Layer).where(Layer.session_id == uuid.UUID(session_id))
    )
    layers = result.scalars().all()
    layer_dicts = []
    for lyr in layers:
        d: dict[str, Any] = {
            "name": lyr.name,
            "type": lyr.layer_type,
            "chunks": [],
            "dataframe": None,
            "geodata": None,
            "image": None,
            "metadata": lyr.metadata_ or {},
            "raw_bytes": b"",
        }
        if lyr.dataframe_json:
            try:
                d["dataframe"] = pd.read_json(
                    io.BytesIO(lyr.dataframe_json.encode()), orient="split"
                )
            except Exception:
                pass
        if lyr.geodata_json:
            try:
                d["geodata"] = json.loads(lyr.geodata_json)
            except Exception:
                pass
        layer_dicts.append(d)
    return layer_dicts


async def _store_output(
    db: AsyncSession,
    session_id: str,
    title: str,
    output_type: str,
    content: Any = None,
    figure_path: str | None = None,
) -> dict:
    out = Output(
        session_id=uuid.UUID(session_id),
        title=title,
        output_type=output_type,
        content_json=json.dumps(content, default=str) if content is not None else None,
        figure_path=figure_path,
    )
    db.add(out)
    await db.flush()
    return {
        "title": title,
        "output_type": output_type,
        "content": content,
        "figure_url": figure_path,
        "timestamp": _ts(),
    }


async def _store_chat(db: AsyncSession, session_id: str, role: str, content: str):
    db.add(ChatMessage(session_id=uuid.UUID(session_id), role=role, content=content))
    await db.flush()


def _find_csv_with_latlon(layers: list[dict]) -> pd.DataFrame | None:
    for lyr in layers:
        df = lyr.get("dataframe")
        if df is not None and "lat" in df.columns and "lon" in df.columns:
            return df
    return None


# ── Dispatch ────────────────────────────────────────────────────────────────

@router.post("/dispatch")
async def dispatch(
    body: DispatchRequest,
    db: AsyncSession = Depends(get_db),
):
    intent, meta = detect_intent(body.prompt)
    await _store_chat(db, body.session_id, "user", body.prompt)

    outputs: list[dict] = []

    if intent == "cross_section":
        outputs = await _handle_cross_section(body.prompt, meta, body.session_id, db)
    elif intent == "hotspot":
        outputs = await _handle_hotspot(body.prompt, meta, body.session_id, db)
    elif intent == "buffer":
        drawn = meta.get("drawn_features", {"type": "FeatureCollection", "features": []})
        outputs = await _handle_buffer(
            body.session_id, meta.get("distance_m", 500.0), drawn, db
        )
    elif intent == "clip":
        drawn = meta.get("drawn_features", {"type": "FeatureCollection", "features": []})
        outputs = await _handle_clip(body.session_id, drawn, db)
    elif intent == "prospectivity":
        outputs = await _handle_prospectivity(body.prompt, body.session_id, db)
    elif intent == "depth_profile":
        outputs = await _handle_depth_profile(body.prompt, meta, body.session_id, db)
    elif intent == "map_display":
        await _store_chat(db, body.session_id, "assistant",
                          "All layers are shown on the map automatically.")
    else:
        outputs = await _handle_qa(body.prompt, body.session_id, db)

    if intent != "qa":
        await _store_chat(db, body.session_id, "assistant",
                          f"Done — **{intent}** result ready.")

    return {"intent": intent, "metadata": meta, "outputs": outputs}


# ── QA ──────────────────────────────────────────────────────────────────────

@router.post("/qa")
async def qa_endpoint(body: QARequest, db: AsyncSession = Depends(get_db)):
    outputs = await _handle_qa(body.prompt, body.session_id, db)
    return {"outputs": outputs}


async def _handle_qa(prompt: str, session_id: str, db: AsyncSession) -> list[dict]:
    hits = await retrieve(prompt, session_id, db, k=5)
    if not hits:
        return [await _store_output(
            db, session_id, "QA - No knowledge base", "text",
            "Upload documents to build the knowledge base.",
        )]

    answer = answer_with_hf(
        prompt, hits,
        model=settings.HF_MODEL, provider=settings.HF_PROVIDER,
    )
    ev_md = "\n\n".join(
        f"**[S{i}] {h['source']} | p.{h['page']}**\n{h['text']}"
        for i, h in enumerate(hits, 1)
    )
    content = f"### Answer\n{answer}\n\n---\n### Evidence\n{ev_md}"
    return [await _store_output(db, session_id, f"QA: {prompt[:55]}", "text", content)]


# ── Cross-Section ───────────────────────────────────────────────────────────

@router.post("/cross-section")
async def cross_section_endpoint(
    body: CrossSectionRequest, db: AsyncSession = Depends(get_db)
):
    meta = {"section_label": body.section_label, "section_length_m": body.section_length_m}
    outputs = await _handle_cross_section(body.prompt, meta, body.session_id, db)
    return {"outputs": outputs}


async def _handle_cross_section(
    prompt: str, meta: dict, session_id: str, db: AsyncSession
) -> list[dict]:
    layers = await _get_session_layers(session_id, db)
    layer_ctx = build_layer_context(layers)
    llm_prompt = build_cross_section_prompt(prompt, layer_ctx)

    interp = generate_from_prompt(
        llm_prompt, model=settings.HF_MODEL, provider=settings.HF_PROVIDER,
        max_new_tokens=900, temperature=0.2,
    )

    combined = prompt + "\n" + interp
    units = parse_units_from_text(combined)
    faults = parse_faults_from_text(combined) or None

    fig = generate_cross_section_figure(
        units=units,
        section_length_m=meta.get("section_length_m", 1000.0),
        section_label=meta.get("section_label", "A — A'"),
        faults=faults,
    )
    fig_url = _save_figure(fig, "Cross_Section_" + meta.get("section_label", "A-A"))

    outputs = []
    outputs.append(await _store_output(
        db, session_id,
        "Cross-Section " + meta.get("section_label", "A-A'"),
        "figure", None, fig_url,
    ))
    outputs.append(await _store_output(
        db, session_id, "Cross-Section Interpretation", "text", interp,
    ))
    return outputs


# ── Hotspot ─────────────────────────────────────────────────────────────────

@router.post("/hotspot")
async def hotspot_endpoint(body: HotspotRequest, db: AsyncSession = Depends(get_db)):
    meta = {"method": body.method, "variable": body.variable}
    outputs = await _handle_hotspot(body.prompt, meta, body.session_id, db)
    return {"outputs": outputs}


async def _handle_hotspot(
    prompt: str, meta: dict, session_id: str, db: AsyncSession
) -> list[dict]:
    layers = await _get_session_layers(session_id, db)
    df = _find_csv_with_latlon(layers)
    if df is None:
        return [await _store_output(
            db, session_id, "Hotspot - No spatial CSV", "text",
            "Upload a CSV with **lat** and **lon** columns first.",
        )]

    var = meta.get("variable", "SnO2")
    method = meta.get("method", "Gi*")
    if var not in df.columns:
        var = next(
            (c for c in df.columns if c not in ("lat", "lon", "hole_id", "INTERVAL")),
            None,
        )
    if var is None:
        return [await _store_output(
            db, session_id, "Hotspot - No numeric column", "text",
            "No suitable numeric column found in the CSV.",
        )]

    try:
        hot_df, hot_meta = hotspot_analysis(
            df.dropna(subset=["lat", "lon"]),
            value_col=var, method=method, k=8,
            grid_cell_m=250.0, z_thresh=1.0,
        )
    except Exception as e:
        return [await _store_output(db, session_id, "Hotspot - Error", "text", str(e))]

    orient = (hot_meta or {}).get("orientation", {})
    lines = [f"**Hotspot ({method} on `{var}`)** - {hot_meta.get('n', len(hot_df))} points"]
    if orient.get("trend_azimuth_deg") is not None:
        lines += [
            f"- Mineralization trend: **{orient['trend_azimuth_deg']:.1f} deg**",
            f"- Recommended trench azimuth: **{orient['trench_azimuth_deg']:.1f} deg**",
        ]
    if hot_meta.get("note"):
        lines.append(f"> {hot_meta['note']}")

    cols = [c for c in ("hole_id", "INTERVAL", "lat", "lon", var, "cluster", "method")
            if c in hot_df.columns]
    table_data = hot_df[cols].head(300).to_dict(orient="records")

    outputs = []
    outputs.append(await _store_output(
        db, session_id, f"Hotspot Table - {var}", "dataframe", table_data,
    ))
    outputs.append(await _store_output(
        db, session_id, f"Hotspot Summary - {var}", "text", "\n".join(lines),
    ))
    return outputs


# ── Buffer ──────────────────────────────────────────────────────────────────

@router.post("/buffer")
async def buffer_endpoint(body: BufferRequest, db: AsyncSession = Depends(get_db)):
    outputs = await _handle_buffer(
        body.session_id, body.distance_m, body.drawn_features, db
    )
    return {"outputs": outputs}


async def _handle_buffer(
    session_id: str, distance_m: float, drawn_features: dict, db: AsyncSession
) -> list[dict]:
    if not drawn_features.get("features"):
        return [await _store_output(
            db, session_id, "Buffer - No geometry", "text",
            "Draw a polygon on the map first.",
        )]

    buffered = buffer_geojson(drawn_features, distance_m)

    # Store as new layer
    buf_layer = Layer(
        session_id=uuid.UUID(session_id),
        name=f"Buffer_{distance_m:.0f}m",
        layer_type="geojson",
        metadata_={"buffer_m": distance_m},
        geodata_json=json.dumps(buffered),
    )
    db.add(buf_layer)
    await db.flush()

    return [await _store_output(
        db, session_id, f"Buffer {distance_m:.0f} m", "geojson", buffered,
    )]


# ── Clip ────────────────────────────────────────────────────────────────────

@router.post("/clip")
async def clip_endpoint(body: ClipRequest, db: AsyncSession = Depends(get_db)):
    outputs = await _handle_clip(body.session_id, body.drawn_features, db)
    return {"outputs": outputs}


async def _handle_clip(
    session_id: str, drawn_features: dict, db: AsyncSession
) -> list[dict]:
    if not drawn_features.get("features"):
        return [await _store_output(
            db, session_id, "Clip - No polygon", "text",
            "Draw a polygon on the map first.",
        )]

    layers = await _get_session_layers(session_id, db)
    df = _find_csv_with_latlon(layers)
    if df is None:
        return [await _store_output(
            db, session_id, "Clip - No spatial CSV", "text",
            "Upload a CSV with lat/lon columns first.",
        )]

    clipped = clip_points_to_polygon(df, drawn_features)

    # Store as new layer
    cl_layer = Layer(
        session_id=uuid.UUID(session_id),
        name="Clipped_points",
        layer_type="csv",
        metadata_={"clipped_rows": len(clipped)},
        dataframe_json=clipped.to_json(orient="split", default_handler=str),
    )
    db.add(cl_layer)
    await db.flush()

    return [await _store_output(
        db, session_id, f"Clip - {len(clipped)} points", "dataframe",
        clipped.head(300).to_dict(orient="records"),
    )]


# ── Prospectivity ───────────────────────────────────────────────────────────

@router.post("/prospectivity")
async def prospectivity_endpoint(
    body: ProspectivityRequest, db: AsyncSession = Depends(get_db)
):
    outputs = await _handle_prospectivity(body.prompt, body.session_id, db)
    return {"outputs": outputs}


async def _handle_prospectivity(
    prompt: str, session_id: str, db: AsyncSession
) -> list[dict]:
    layers = await _get_session_layers(session_id, db)
    csv_lyrs = [l for l in layers if l["type"] == "csv" and l["dataframe"] is not None]
    if not csv_lyrs:
        return [await _store_output(
            db, session_id, "Prospectivity - No XRF CSV", "text",
            "Upload an XRF CSV first.",
        )]

    xrf_raw = csv_lyrs[0]["dataframe"]
    try:
        xrf_df = ensure_oxides(xrf_raw.copy(), prefer_existing_oxides=True,
                               allow_element_to_oxide=True)
        xrf_df = add_depth_interval_fields(xrf_df, pit_depth_col="depth")
        xrf_df["FracProxy"] = (
            pd.to_numeric(xrf_df.get("Rb2O", 0), errors="coerce").fillna(0)
            + pd.to_numeric(xrf_df.get("Cs2O", 0), errors="coerce").fillna(0)
        ) / (pd.to_numeric(xrf_df.get("SrO", 0), errors="coerce").fillna(0) + 1e-6)

        xrf_sum = summarize_xrf(xrf_df)
        rating, sigs = quick_prospectivity(xrf_sum)

        evidence = await retrieve(prompt, session_id, db, k=6)

        ml_bundle = None
        try:
            feat_df = build_features(xrf_df)
            sn = pd.to_numeric(feat_df.get("SnO2", np.nan), errors="coerce")
            q1, q2 = sn.quantile(0.33), sn.quantile(0.66)
            feat_df["prospectivity_class"] = sn.apply(
                lambda v: "Low" if (pd.isna(v) or v <= q1)
                else ("Medium" if v <= q2 else "High")
            )
            ml_bundle = train_baselines(feat_df, target_col="prospectivity_class")
        except Exception:
            pass

        preds = None
        if "hole_id" in xrf_df.columns:
            preds = within_hole_prediction(xrf_df, hole_col="hole_id", value_col="SnO2")

    except Exception as e:
        return [await _store_output(
            db, session_id, "Prospectivity - Error", "text", str(e)
        )]

    # Generate narrative
    narr_prompt = build_snree_prompt(
        question=prompt, xrf_summary=xrf_sum,
        evidence=evidence, prospectivity=rating, signals=sigs,
    )
    narrative = generate_from_prompt(
        narr_prompt, model=settings.HF_MODEL, provider=settings.HF_PROVIDER,
        max_new_tokens=700, temperature=0.2,
    )

    ml_txt = ""
    if ml_bundle:
        for nm, info in ml_bundle["models"].items():
            ml_txt += (f"\n**{nm}** - accuracy: {info.get('accuracy', 0):.2%}, "
                       f"features: {info.get('features', [])}")

    result_md = (
        f"## Sn-REE Prospectivity: **{rating}**\n\n"
        f"### Geochemical Signals\n```json\n{json.dumps(sigs, indent=2, default=str)}\n```\n\n"
        f"### Geological Narrative\n{narrative}"
        + (f"\n\n### ML Baselines\n{ml_txt}" if ml_txt else "")
    )

    outputs = [await _store_output(
        db, session_id, f"Prospectivity - {rating}", "text", result_md,
    )]

    # Depth profiles
    if preds:
        for hole, p in list(preds.items())[:4]:
            outputs.append(await _store_output(
                db, session_id, f"Depth Profile - {hole}", "dataframe",
                {"measured_depth_m": p["measured_depth_m"],
                 "measured_value": p["measured_value"],
                 "depth_grid_m": p["depth_grid_m"],
                 "predicted": p["predicted"]},
            ))

    return outputs


# ── Depth Profile ───────────────────────────────────────────────────────────

@router.post("/depth-profile")
async def depth_profile_endpoint(
    body: DepthProfileRequest, db: AsyncSession = Depends(get_db)
):
    meta = {"variable": body.variable}
    outputs = await _handle_depth_profile(body.prompt, meta, body.session_id, db)
    return {"outputs": outputs}


async def _handle_depth_profile(
    prompt: str, meta: dict, session_id: str, db: AsyncSession
) -> list[dict]:
    layers = await _get_session_layers(session_id, db)
    csv_lyrs = [l for l in layers if l["type"] == "csv" and l["dataframe"] is not None]
    if not csv_lyrs:
        return [await _store_output(
            db, session_id, "Depth Profile - No CSV", "text",
            "Upload an assay CSV with a hole_id column.",
        )]

    df = csv_lyrs[0]["dataframe"]
    var = meta.get("variable", "SnO2")
    if var not in df.columns:
        var = next(
            (c for c in df.columns if c not in ("lat", "lon", "hole_id", "INTERVAL")),
            "SnO2",
        )
    if "hole_id" not in df.columns:
        return [await _store_output(
            db, session_id, "Depth Profile - No hole_id", "text",
            "CSV needs a `hole_id` column.",
        )]

    preds = within_hole_prediction(df, hole_col="hole_id", value_col=var)

    outputs = []
    for hole, p in list(preds.items())[:4]:
        outputs.append(await _store_output(
            db, session_id, f"Depth Profile - {hole} ({var})", "dataframe",
            {"measured_depth_m": p["measured_depth_m"],
             "measured_value": p["measured_value"],
             "depth_grid_m": p["depth_grid_m"],
             "predicted": p["predicted"],
             "max_depth_m": p["max_depth_m"]},
        ))
    return outputs
