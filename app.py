# app.py  -  GeoMinerAI  |  Single-workspace edition
# ============================================================
#  Layout (left → right, top → bottom on right side)
#  ┌──────────────┬─────────────────────────────────────────┐
#  │              │  MAP LAYOUT  (analysis outputs)         │
#  │  LAYER LIST  ├─────────────────────────────────────────┤
#  │              │  LIVE MAP  (interactive, draw tools)    │
#  │              ├─────────────────────────────────────────┤
#  │              │  PROMPT BOX  (NLP + file upload)        │
#  └──────────────┴─────────────────────────────────────────┘
# ============================================================

from __future__ import annotations

import io
import json
import math
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import folium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from folium.plugins import Draw, MeasureControl, MousePosition
from streamlit_folium import st_folium

from core.chemistry import ensure_oxides
from core.cross_section import (
    build_cross_section_prompt,
    build_layer_context,
    generate_cross_section_figure,
    parse_faults_from_text,
    parse_units_from_text,
)
from core.geo_ingest import (
    build_index_from_layers,
    get_csv_layers,
    ingest_file,
)
from core.geoprocessing import (
    buffer_geojson,
    clip_points_to_polygon,
    drawn_to_geojson,
    nearest_row,
)
from core.intent import detect_intent
from core.llm_hf import answer_with_hf, generate_from_prompt
from core.rag import retrieve
from core.snree_hotspots import hotspot_analysis
from core.snree_lab import (
    add_depth_interval_fields,
    build_snree_prompt,
    quick_prospectivity,
    summarize_xrf,
)
from core.snree_ml import build_features, train_baselines, within_hole_prediction

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GeoMinerAI",
    layout="wide",
    initial_sidebar_state="collapsed",
    page_icon="🌍",
)

st.markdown(
    """
<style>
.block-container {padding-top:0.8rem; padding-bottom:1rem;}
[data-testid="stVerticalBlock"] > div {padding-top:0; padding-bottom:0;}
</style>
""",
    unsafe_allow_html=True,
)

# ── session state ─────────────────────────────────────────────────────────────
_DEFAULTS: Dict[str, Any] = {
    "layers":          [],
    "index":           None,
    "chunks":          [],
    "drawn_features":  {"type": "FeatureCollection", "features": []},
    "map_center":      [9.9, 8.9],
    "map_zoom":        8,
    "outputs":         [],
    "chat":            [],
    "snree":           {
        "xrf_df": None, "xrf_summary": None, "rating": None,
        "signals": None, "hot_df": None, "hot_meta": None,
        "evidence": [], "ml_bundle": None, "preds": None,
        "last_run_ok": False,
    },
    "selected_hole": None,
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    hf_ok = False
    try:
        hf_ok = bool(
            os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
            or st.secrets.get("HF_TOKEN", "")
        )
    except Exception:
        pass
    st.write("HF token:", "✅ found" if hf_ok else "❌ missing")
    st.caption("Set HF_TOKEN in .streamlit/secrets.toml or as env var.")

    st.divider()
    provider = st.selectbox("Provider", ["auto", "hf-inference"], index=0, key="provider_sel")
    hf_model = st.selectbox(
        "LLM model",
        [
            "HuggingFaceH4/zephyr-7b-beta",
            "mistralai/Mistral-7B-Instruct-v0.3",
            "microsoft/phi-2",
            "google/gemma-2b-it",
            "tiiuae/falcon-7b-instruct",
        ],
        index=0, key="model_sel",
    )
    if st.button("Test LLM"):
        st.write(answer_with_hf(
            "Explain greisen alteration in one sentence.",
            evidence=[], model=hf_model, provider=provider,
        ))

    st.divider()
    st.subheader("Map defaults")
    d_lat  = st.number_input("Centre lat",  value=9.9,  step=0.1, key="d_lat")
    d_lon  = st.number_input("Centre lon",  value=8.9,  step=0.1, key="d_lon")
    d_zoom = st.slider("Zoom", 2, 18, 8, key="d_zoom")
    if st.button("Apply defaults"):
        st.session_state.map_center = [d_lat, d_lon]
        st.session_state.map_zoom   = d_zoom
        st.rerun()

    st.divider()
    if st.button("🗑 Clear outputs"):
        st.session_state.outputs = []
        st.session_state.chat    = []
        st.rerun()
    if st.button("🗑 Clear all layers"):
        st.session_state.layers  = []
        st.session_state.index   = None
        st.session_state.chunks  = []
        st.rerun()


# ── helpers ───────────────────────────────────────────────────────────────────
def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _push_output(title: str, otype: str, content: Any,
                 layer: Optional[Dict] = None) -> None:
    st.session_state.outputs.insert(0, {
        "title": title, "type": otype,
        "content": content, "layer": layer, "ts": _ts(),
    })


def _push_chat(role: str, content: str) -> None:
    st.session_state.chat.append({"role": role, "content": content})


def _rebuild_index() -> None:
    idx, chunks = build_index_from_layers(st.session_state.layers)
    st.session_state.index  = idx
    st.session_state.chunks = chunks


def _find_csv_with_latlon() -> Optional[pd.DataFrame]:
    for lyr in st.session_state.layers:
        df = lyr.get("dataframe")
        if df is not None and "lat" in df.columns and "lon" in df.columns:
            return df
    return None


def _color_cluster(c: str) -> str:
    c = str(c).upper()
    if any(k in c for k in ("HH", "HOT", "HIGH")):  return "red"
    if any(k in c for k in ("LL", "COLD", "LOW")):   return "blue"
    if "MED" in c:                                    return "orange"
    return "gray"


# ── handlers ──────────────────────────────────────────────────────────────────
def handle_qa(prompt: str) -> None:
    hf_model = st.session_state.get("model_sel", "HuggingFaceH4/zephyr-7b-beta")
    provider  = st.session_state.get("provider_sel", "auto")
    if st.session_state.index is None:
        _push_output("QA - No knowledge base", "text",
                     "Upload documents in the **Prompt Box** to build the knowledge base.")
        return
    hits   = retrieve(prompt, st.session_state.index, st.session_state.chunks, k=5)
    answer = answer_with_hf(prompt, hits, model=hf_model, provider=provider)
    ev_md  = "\n\n".join(
        f"**[S{i}] {h['source']} | p.{h['page']}**\n{h['text']}"
        for i, h in enumerate(hits, 1)
    )
    _push_output(f"QA: {prompt[:55]}", "text",
                 f"### Answer\n{answer}\n\n---\n### Evidence\n{ev_md}")


def handle_cross_section(prompt: str, meta: Dict) -> None:
    hf_model = st.session_state.get("model_sel", "HuggingFaceH4/zephyr-7b-beta")
    provider  = st.session_state.get("provider_sel", "auto")

    layer_ctx = build_layer_context(st.session_state.layers)
    drawn = st.session_state.drawn_features
    if drawn.get("features"):
        layer_ctx += (
            "\n\nUser-drawn section trace (GeoJSON):\n"
            + json.dumps(drawn, indent=2)[:800]
        )

    llm_prompt = build_cross_section_prompt(prompt, layer_ctx)
    with st.spinner("Constructing geological cross-section …"):
        interp = generate_from_prompt(
            llm_prompt, model=hf_model, provider=provider,
            max_new_tokens=900, temperature=0.2,
        )

    combined = prompt + "\n" + interp
    units    = parse_units_from_text(combined)
    faults   = parse_faults_from_text(combined) or None

    fig = generate_cross_section_figure(
        units=units,
        section_length_m=meta.get("section_length_m", 1000.0),
        section_label=meta.get("section_label", "A - A'"),
        faults=faults,
    )
    _push_output("Cross-Section " + meta.get('section_label','A-A prime'), "figure", fig)
    _push_output("Cross-Section Interpretation", "text", interp)


def handle_hotspot(prompt: str, meta: Dict) -> None:
    df = _find_csv_with_latlon()
    if df is None:
        _push_output("Hotspot - No spatial CSV", "text",
                     "Upload a CSV with **lat** and **lon** columns first.")
        return

    var    = meta.get("variable", "SnO2")
    method = meta.get("method",   "Gi*")
    if var not in df.columns:
        var = next((c for c in df.columns
                    if c not in ("lat", "lon", "hole_id", "INTERVAL")), None)
    if var is None:
        _push_output("Hotspot - No numeric column", "text",
                     "No suitable numeric column found in the CSV.")
        return

    with st.spinner(f"Running {method} hotspot analysis on {var} …"):
        try:
            hot_df, hot_meta = hotspot_analysis(
                df.dropna(subset=["lat", "lon"]),
                value_col=var, method=method, k=8,
                grid_cell_m=250.0, z_thresh=1.0,
            )
        except Exception as e:
            _push_output("Hotspot - Error", "text", f"{e}")
            return

    st.session_state.snree["hot_df"]   = hot_df
    st.session_state.snree["hot_meta"] = hot_meta

    orient = (hot_meta or {}).get("orientation", {})
    lines  = [f"**Hotspot ({method} on `{var}`)** - {hot_meta.get('n', len(hot_df))} points"]
    if orient.get("trend_azimuth_deg") is not None:
        lines += [
            f"- Mineralization trend: **{orient['trend_azimuth_deg']:.1f}°**",
            f"- Recommended trench azimuth: **{orient['trench_azimuth_deg']:.1f}°**",
        ]
    if hot_meta.get("note"):
        lines.append(f"> {hot_meta['note']}")

    cols = [c for c in ("hole_id", "INTERVAL", "lat", "lon",
                         var, "cluster", "method") if c in hot_df.columns]
    _push_output(f"Hotspot Table - {var}", "dataframe", hot_df[cols].head(300))
    _push_output(f"Hotspot Summary - {var}", "text", "\n".join(lines))

    lats = pd.to_numeric(hot_df["lat"], errors="coerce").dropna()
    lons = pd.to_numeric(hot_df["lon"], errors="coerce").dropna()
    if len(lats):
        st.session_state.map_center = [float(lats.mean()), float(lons.mean())]
        st.session_state.map_zoom   = 12


def handle_buffer(prompt: str, meta: Dict) -> None:
    drawn = st.session_state.drawn_features
    if not drawn.get("features"):
        _push_output("Buffer - No geometry", "text",
                     "Draw a polygon on the **Live Map** first.")
        return
    dist_m = meta.get("distance_m", 500.0)
    with st.spinner(f"Buffering by {dist_m:.0f} m …"):
        buffered = buffer_geojson(drawn, dist_m)
    buf_layer = {
        "name": f"Buffer_{dist_m:.0f}m", "type": "geojson",
        "chunks": [], "dataframe": None, "geodata": buffered,
        "image": None, "metadata": {"buffer_m": dist_m}, "raw_bytes": b"",
    }
    st.session_state.layers.append(buf_layer)
    _rebuild_index()
    _push_output(f"Buffer {dist_m:.0f} m", "geojson", buffered, layer=buf_layer)


def handle_clip(prompt: str) -> None:
    drawn = st.session_state.drawn_features
    if not drawn.get("features"):
        _push_output("Clip - No polygon", "text",
                     "Draw a polygon on the **Live Map** first.")
        return
    df = _find_csv_with_latlon()
    if df is None:
        _push_output("Clip - No spatial CSV", "text",
                     "Upload a CSV with lat/lon columns first.")
        return
    with st.spinner("Clipping …"):
        clipped = clip_points_to_polygon(df, drawn)
    cl = {
        "name": "Clipped_points", "type": "csv",
        "chunks": [], "dataframe": clipped, "geodata": None,
        "image": None, "metadata": {"clipped_rows": len(clipped)}, "raw_bytes": b"",
    }
    st.session_state.layers.append(cl)
    _rebuild_index()
    _push_output(f"Clip - {len(clipped)} points", "dataframe", clipped, layer=cl)


def handle_prospectivity(prompt: str) -> None:
    hf_model = st.session_state.get("model_sel", "HuggingFaceH4/zephyr-7b-beta")
    provider  = st.session_state.get("provider_sel", "auto")

    csv_lyrs = get_csv_layers(st.session_state.layers)
    if not csv_lyrs:
        _push_output("Prospectivity - No XRF CSV", "text",
                     "Upload an XRF CSV first.")
        return

    xrf_raw = csv_lyrs[0]["dataframe"]
    with st.spinner("Running Sn-REE prospectivity …"):
        try:
            xrf_df = ensure_oxides(xrf_raw.copy(),
                                   prefer_existing_oxides=True,
                                   allow_element_to_oxide=True)
            xrf_df = add_depth_interval_fields(xrf_df, pit_depth_col="depth")
            xrf_df["FracProxy"] = (
                pd.to_numeric(xrf_df.get("Rb2O", 0), errors="coerce").fillna(0)
                + pd.to_numeric(xrf_df.get("Cs2O", 0), errors="coerce").fillna(0)
            ) / (pd.to_numeric(xrf_df.get("SrO", 0), errors="coerce").fillna(0) + 1e-6)

            xrf_sum       = summarize_xrf(xrf_df)
            rating, sigs  = quick_prospectivity(xrf_sum)

            evidence = []
            if st.session_state.index is not None:
                evidence = retrieve(prompt, st.session_state.index,
                                    st.session_state.chunks, k=6)

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
                preds = within_hole_prediction(xrf_df, hole_col="hole_id",
                                               value_col="SnO2")

            st.session_state.snree.update({
                "xrf_df": xrf_df, "xrf_summary": xrf_sum,
                "rating": rating, "signals": sigs,
                "evidence": evidence, "ml_bundle": ml_bundle,
                "preds": preds, "last_run_ok": True,
            })

            if "lat" in xrf_df.columns and "lon" in xrf_df.columns:
                hot_df, hot_meta = hotspot_analysis(
                    xrf_df.dropna(subset=["lat", "lon"]),
                    value_col="SnO2", method="Gi*",
                )
                st.session_state.snree["hot_df"]   = hot_df
                st.session_state.snree["hot_meta"] = hot_meta

        except Exception as e:
            _push_output("Prospectivity - Error", "text", str(e))
            return

    narr_prompt = build_snree_prompt(
        question=prompt, xrf_summary=xrf_sum,
        evidence=evidence, prospectivity=rating, signals=sigs,
    )
    narrative = generate_from_prompt(
        narr_prompt, model=hf_model, provider=provider,
        max_new_tokens=700, temperature=0.2,
    )

    ml_txt = ""
    if ml_bundle:
        for nm, info in ml_bundle["models"].items():
            ml_txt += (f"\n**{nm}** - accuracy: {info.get('accuracy', 0):.2%}, "
                       f"features: {info.get('features', [])}")

    result_md = (
        f"## Sn-REE Prospectivity: **{rating}**\n\n"
        f"### Geochemical Signals\n```json\n{json.dumps(sigs, indent=2)}\n```\n\n"
        f"### Geological Narrative\n{narrative}"
        + (f"\n\n### ML Baselines\n{ml_txt}" if ml_txt else "")
    )
    _push_output(f"Prospectivity - {rating}", "text", result_md)

    if preds:
        _render_depth_profiles(preds, "SnO2")


def handle_depth_profile(prompt: str, meta: Dict) -> None:
    csv_lyrs = get_csv_layers(st.session_state.layers)
    if not csv_lyrs:
        _push_output("Depth Profile - No CSV", "text",
                     "Upload an assay CSV with a hole_id column.")
        return
    df  = csv_lyrs[0]["dataframe"]
    var = meta.get("variable", "SnO2")
    if var not in df.columns:
        var = next((c for c in df.columns
                    if c not in ("lat", "lon", "hole_id", "INTERVAL")), "SnO2")
    if "hole_id" not in df.columns:
        _push_output("Depth Profile - No hole_id", "text",
                     "CSV needs a `hole_id` column.")
        return
    preds = within_hole_prediction(df, hole_col="hole_id", value_col=var)
    _render_depth_profiles(preds, var)


def _render_depth_profiles(preds: Dict, var: str) -> None:
    from matplotlib import gridspec
    _IV_COLORS = {"A": "#f9e4b7", "B": "#d5e8d4", "C": "#dae8fc", "D": "#e1d5e7"}
    for hole, p in list(preds.items())[:4]:
        fig = plt.figure(figsize=(10, 5))
        gs  = gridspec.GridSpec(1, 2, width_ratios=[3, 1], wspace=0.3)
        ax  = fig.add_subplot(gs[0])
        axL = fig.add_subplot(gs[1])

        meas = pd.DataFrame({
            "depth": p["measured_depth_m"],
            "val":   p["measured_value"],
            "iv":    p.get("measured_interval",
                           ["?"] * len(p["measured_depth_m"])),
        }).sort_values("depth")

        ax.plot(p["predicted"], p["depth_grid_m"],
                lw=2, color="#1a6fa8", label="Trend")
        ax.scatter(meas["val"], meas["depth"],
                   s=55, zorder=5, color="#e67e22", label="Measured")
        ax.set_xlabel(var)
        ax.set_ylabel("Depth (m)")
        ax.set_title(f"Hole: {hole}")
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

        bands = [("A", 0.0, 0.3), ("B", 0.3, 1.0),
                 ("C", 1.0, 2.0), ("D", 2.0, float(p["max_depth_m"]))]
        for nm, y0, y1 in bands:
            if y1 > y0:
                axL.axhspan(y0, y1, color=_IV_COLORS.get(nm, "#eee"), alpha=0.6)
                axL.text(0.5, (y0 + y1) / 2, nm,
                         ha="center", va="center", fontsize=10,
                         fontweight="bold", color="#333")
        axL.set_xlim(0, 1)
        axL.set_xticks([])
        axL.set_ylabel("Depth (m)")
        axL.set_title("Interval log")
        axL.invert_yaxis()
        plt.tight_layout()
        _push_output(f"Depth Profile - {hole} ({var})", "figure", fig)


# ── central dispatcher ────────────────────────────────────────────────────────
def process_prompt(user_text: str, uploaded_files) -> None:
    hf_model = st.session_state.get("model_sel", "HuggingFaceH4/zephyr-7b-beta")
    provider  = st.session_state.get("provider_sel", "auto")

    # Ingest uploaded files
    new_names = []
    if uploaded_files:
        for uf in uploaded_files:
            if any(l["name"] == uf.name for l in st.session_state.layers):
                continue
            with st.spinner(f"Ingesting {uf.name} …"):
                layer = ingest_file(uf)
            st.session_state.layers.append(layer)
            new_names.append(uf.name)
        if new_names:
            _rebuild_index()
            _push_chat(
                "assistant",
                f"✅ Loaded: {', '.join(new_names)}. "
                f"Knowledge base: {len(st.session_state.chunks)} chunks.",
            )

    text = user_text.strip()
    if not text:
        return

    _push_chat("user", text)
    intent, meta = detect_intent(text)

    if intent == "cross_section":
        handle_cross_section(text, meta)
    elif intent == "hotspot":
        handle_hotspot(text, meta)
    elif intent == "buffer":
        handle_buffer(text, meta)
    elif intent == "clip":
        handle_clip(text)
    elif intent == "prospectivity":
        handle_prospectivity(text)
    elif intent == "depth_profile":
        handle_depth_profile(text, meta)
    elif intent == "map_display":
        _push_chat("assistant",
                   "All layers are shown on the **Live Map** automatically.")
    else:
        handle_qa(text)

    if intent != "qa":
        _push_chat("assistant",
                   f"Done - **{intent}** result pushed to Map Layout ↑")


# ── panel renderers ───────────────────────────────────────────────────────────
def render_layer_list() -> None:
    st.markdown("### 🗂 Layers")
    layers = st.session_state.layers
    if not layers:
        st.caption("No layers yet.\nUpload files in the Prompt Box ↓")
        return

    _ICONS = {
        "pdf": "📄", "word": "📝", "csv": "📊", "geojson": "🗾",
        "kml": "🗺", "raster": "🛰", "image": "🖼",
        "vector": "🔷", "text": "📃", "unknown": "❓",
    }
    for i, lyr in enumerate(layers):
        icon = _ICONS.get(lyr["type"], "❓")
        with st.expander(f"{icon} {lyr['name']}", expanded=False):
            meta = lyr.get("metadata", {})
            st.write(f"**Type:** {lyr['type']}")
            kb = meta.get("size_bytes", 0)
            if kb:
                st.write(f"**Size:** {kb/1024:.1f} KB")
            df = lyr.get("dataframe")
            if df is not None:
                st.write(f"**Columns:** {list(df.columns)}")
                st.write(f"**Rows:** {len(df)}")
                has_geo = "lat" in df.columns and "lon" in df.columns
                st.write(f"**Spatial:** {'yes' if has_geo else 'no'}")
            gd = lyr.get("geodata")
            if gd:
                st.write(f"**Features:** {len(gd.get('features', []))}")
            if meta.get("crs"):
                st.write(f"**CRS:** {meta['crs']}")
            if meta.get("note"):
                st.caption(meta["note"])
            if st.button("🗑 Remove", key=f"rm_{i}"):
                st.session_state.layers.pop(i)
                _rebuild_index()
                st.rerun()

    st.divider()
    st.caption(f"**{len(layers)}** layer(s)")
    if st.session_state.index is not None:
        st.caption(f"Index: {len(st.session_state.chunks)} chunks")


def render_map_layout() -> None:
    outputs = st.session_state.outputs
    if not outputs:
        st.info(
            "Analysis outputs appear here automatically. "
            "Use the Prompt Box below - try: "
            "run hotspot analysis | draw cross section | run prospectivity analysis"
        )
        return

    for i, out in enumerate(outputs):
        label = f"[{out['ts']}] {out['title']}"
        with st.expander(label, expanded=(i == 0)):
            otype, content = out["type"], out["content"]

            if otype == "text":
                st.markdown(content)

            elif otype == "figure":
                st.pyplot(content, clear_figure=False)
                buf = io.BytesIO()
                content.savefig(buf, format="png", dpi=150, bbox_inches="tight")
                st.download_button(
                    "⬇ Download PNG", buf.getvalue(),
                    f"{out['title'].replace(' ','_')}.png", "image/png",
                    key=f"dl_fig_{i}",
                )

            elif otype == "dataframe":
                st.dataframe(content, use_container_width=True)
                st.download_button(
                    "⬇ Download CSV",
                    content.to_csv(index=False).encode(),
                    f"{out['title'].replace(' ','_')}.csv", "text/csv",
                    key=f"dl_df_{i}",
                )

            elif otype == "geojson":
                st.json(content, expanded=False)
                st.download_button(
                    "⬇ Download GeoJSON",
                    json.dumps(content, indent=2).encode(),
                    f"{out['title'].replace(' ','_')}.geojson",
                    "application/geo+json",
                    key=f"dl_gj_{i}",
                )

            elif otype == "image":
                st.image(content)


def render_live_map() -> None:
    center = st.session_state.map_center
    zoom   = st.session_state.map_zoom

    m = folium.Map(location=center, zoom_start=zoom,
                   control_scale=True, tiles="OpenStreetMap")

    Draw(
        export=False, position="topleft",
        draw_options={
            "polyline":     {"shapeOptions": {"color": "#e74c3c"}},
            "polygon":      {"shapeOptions": {"color": "#2980b9",
                                              "fillOpacity": 0.15}},
            "rectangle":    {"shapeOptions": {"color": "#27ae60",
                                              "fillOpacity": 0.10}},
            "circle":       False,
            "marker":       True,
            "circlemarker": False,
        },
    ).add_to(m)

    MeasureControl(primary_length_unit="meters",
                   secondary_length_unit="kilometers").add_to(m)
    MousePosition(position="bottomleft").add_to(m)

    # Restore saved drawings
    saved = st.session_state.drawn_features
    if saved.get("features"):
        folium.GeoJson(
            saved, name="Saved drawings",
            style_function=lambda _: {"color": "#2980b9", "fillColor": "#2980b9",
                                      "weight": 2, "fillOpacity": 0.1},
        ).add_to(m)

    hot_df = st.session_state.snree.get("hot_df")

    # CSV point layers
    for lyr in st.session_state.layers:
        df = lyr.get("dataframe")
        if df is None or "lat" not in df.columns or "lon" not in df.columns:
            continue
        pdf = df.copy()
        pdf["lat"] = pd.to_numeric(pdf["lat"], errors="coerce")
        pdf["lon"] = pd.to_numeric(pdf["lon"], errors="coerce")
        pdf = pdf.dropna(subset=["lat", "lon"])
        if pdf.empty:
            continue

        use_clusters = (
            hot_df is not None
            and "cluster" in hot_df.columns
            and len(hot_df) == len(df)
        )
        src = hot_df if use_clusters else pdf
        fg  = folium.FeatureGroup(name=lyr["name"], show=True)

        for _, r in src.iterrows():
            try:
                la, lo = float(r["lat"]), float(r["lon"])
            except Exception:
                continue
            cl     = str(r.get("cluster", "")) if use_clusters else ""
            colour = _color_cluster(cl) if use_clusters else "#3498db"
            hid    = str(r.get("hole_id", ""))
            popup  = f"<b>{hid}</b><br>Cluster: {cl}" if use_clusters else hid
            folium.CircleMarker(
                [la, lo], radius=6,
                color=colour, fill=True, fill_color=colour,
                fill_opacity=0.85,
                popup=folium.Popup(popup, max_width=220),
                tooltip=hid or f"{la:.4f},{lo:.4f}",
            ).add_to(fg)
        fg.add_to(m)

    # GeoJSON layers
    for lyr in st.session_state.layers:
        gd = lyr.get("geodata")
        if gd is None:
            continue
        folium.GeoJson(
            gd, name=lyr["name"],
            style_function=lambda _: {"color": "#8e44ad", "fillColor": "#8e44ad",
                                      "weight": 2, "fillOpacity": 0.1},
            tooltip=lyr["name"],
        ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    map_state = st_folium(
        m, key="main_map", height=380, width=None,
        returned_objects=["all_drawings", "last_clicked"],
    )

    # Capture drawings
    raw = (map_state or {}).get("all_drawings")
    if raw:
        fc = drawn_to_geojson(raw)
        if fc.get("features"):
            st.session_state.drawn_features = fc
            st.caption(f"✏️ {len(fc['features'])} shape(s) drawn.")

    # Map-click selection
    clicked = (map_state or {}).get("last_clicked")
    if clicked and "lat" in clicked and "lng" in clicked:
        df_geo = _find_csv_with_latlon()
        if df_geo is not None and not df_geo.empty:
            try:
                row, dist = nearest_row(
                    df_geo.dropna(subset=["lat", "lon"]),
                    float(clicked["lat"]), float(clicked["lng"]),
                )
                hid = str(row.get("hole_id", "?"))
                if hid != st.session_state.selected_hole:
                    st.session_state.selected_hole = hid
                    st.toast(f"📍 Selected: {hid}  ({dist:.0f} m)")
            except Exception:
                pass

    c1, c2 = st.columns(2)
    with c1:
        if st.button("💾 Save drawings → Layer list", key="save_draw"):
            d = st.session_state.drawn_features
            if d.get("features"):
                dl = {
                    "name": f"Drawing_{_ts().replace(':','')}",
                    "type": "geojson", "chunks": [], "dataframe": None,
                    "geodata": d, "image": None,
                    "metadata": {"source": "map drawing",
                                 "features": len(d["features"])},
                    "raw_bytes": b"",
                }
                st.session_state.layers.append(dl)
                _rebuild_index()
                st.success("Drawing saved as a layer.")
            else:
                st.warning("Nothing drawn yet.")
    with c2:
        if st.button("🗑 Clear drawings", key="clr_draw"):
            st.session_state.drawn_features = {
                "type": "FeatureCollection", "features": []
            }
            st.rerun()


def render_prompt_box() -> None:
    st.markdown("#### 🧠 Prompt Box")

    for msg in st.session_state.chat[-8:]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    uploaded = st.file_uploader(
        "📂 Upload files (PDF, Word, CSV, GeoJSON, images, rasters …)",
        type=[
            "pdf", "docx", "doc", "csv",
            "geojson", "json", "kml",
            "tif", "tiff", "dem", "asc",
            "png", "jpg", "jpeg", "bmp",
            "shp", "gpkg", "txt", "md",
        ],
        accept_multiple_files=True,
        key="kb_uploader",
        label_visibility="collapsed",
    )

    # Quick-command chips
    st.caption("Quick commands →")
    qc1, qc2, qc3, qc4 = st.columns(4)
    if qc1.button("🔥 Hotspot",       key="qc_hs"):
        process_prompt("Run hotspot analysis on the uploaded CSV", [])
        st.rerun()
    if qc2.button("📐 Cross-Section", key="qc_xs"):
        process_prompt("Draw a geological cross section A to A'", [])
        st.rerun()
    if qc3.button("💎 Prospectivity", key="qc_pr"):
        process_prompt("Run Sn-REE prospectivity analysis", [])
        st.rerun()
    if qc4.button("📏 Buffer 500m",   key="qc_buf"):
        process_prompt("Buffer drawn polygon by 500 m", [])
        st.rerun()

    user_input = st.chat_input(
        "Ask or command … "
        "(e.g. draw cross section A-A prime | run hotspot on SnO2 | buffer 1km)",
        key="chat_input",
    )

    if user_input:
        process_prompt(user_input, uploaded)
        st.rerun()
    elif uploaded:
        process_prompt("", uploaded)
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# ASSEMBLE PAGE
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## 🌍 GeoMinerAI - Geological Exploration Workspace")
st.caption(
    "RAG · Hotspot Analysis · ML Prospectivity · "
    "Geological Cross-Section · Geoprocessing · Multi-format Knowledge Base"
)

hf_model = st.session_state.get("model_sel",    "HuggingFaceH4/zephyr-7b-beta")
provider  = st.session_state.get("provider_sel", "auto")

col_left, col_right = st.columns([1, 3], gap="small")

with col_left:
    render_layer_list()

with col_right:
    st.markdown("#### 🗺 Map Layout - Analysis Outputs")
    render_map_layout()
    st.divider()
    st.markdown("#### 🌐 Live Map")
    render_live_map()
    st.divider()
    render_prompt_box()
