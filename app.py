# app.py
import os
import json
import math

import streamlit as st
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import gridspec

import folium
from streamlit_folium import st_folium

from sklearn.model_selection import train_test_split

from core.rag import build_index_from_pdfs, retrieve
from core.llm_hf import answer_with_hf
from core.geogpt import load_geogpt_csv
from core.benchmark import run_closed_book_benchmark

from core.chemistry import ensure_oxides
from core.snree_hotspots import hotspot_analysis
from core.snree_lab import (
    load_xrf_csv,
    add_depth_interval_fields,
    summarize_xrf,
    quick_prospectivity,
    build_snree_prompt,
)
from core.snree_ml import build_features, train_baselines, within_hole_prediction


# -----------------------------
# Small GIS helpers
# -----------------------------
def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dl / 2.0) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def nearest_row_by_click(df, click_lat, click_lon, lat_col="lat", lon_col="lon"):
    dists = df.apply(
        lambda r: haversine_m(click_lat, click_lon, float(r[lat_col]), float(r[lon_col])),
        axis=1,
    )
    i = int(dists.idxmin())
    return df.loc[i], float(dists.loc[i])


# -----------------------------
# App shell
# -----------------------------
st.set_page_config(page_title="GeoMinerAI", layout="wide")

st.title("GeoMinerAI")
st.caption("Upload geoscience PDFs, retrieve evidence, and answer with citations using a Hugging Face model.")

# Session state
if "index" not in st.session_state:
    st.session_state.index = None
    st.session_state.chunks = None

if "selected_hole_from_map" not in st.session_state:
    st.session_state.selected_hole_from_map = None

if "striplog_notes" not in st.session_state:
    st.session_state.striplog_notes = {
        "A": {"lithology": "", "alteration": ""},
        "B": {"lithology": "", "alteration": ""},
        "C": {"lithology": "", "alteration": ""},
        "D": {"lithology": "", "alteration": ""},
    }

# This is the critical persistence bundle for Sn–REE Lab outputs
if "snree" not in st.session_state:
    st.session_state.snree = {
        "xrf_df": None,
        "xrf_summary": None,
        "rating": None,
        "signals": None,
        "hot_df": None,
        "hot_meta": None,
        "evidence": None,
        "ml_bundle": None,
        "preds": None,
        "last_run_ok": False,
        "last_run_params": None,
    }


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("Token check")

    hf_from_env = bool(os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN"))
    hf_from_secrets = False
    try:
        hf_from_secrets = bool(st.secrets.get("HF_TOKEN", ""))
    except Exception:
        hf_from_secrets = False

    st.write("HF token found:", "Yes" if (hf_from_env or hf_from_secrets) else "No")
    st.caption("Local: .streamlit/secrets.toml. Online: set HF_TOKEN in Streamlit Cloud Secrets.")

    st.divider()
    st.header("PDF knowledge base (RAG)")
    pdfs = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

    if st.button("Build PDF Index"):
        if not pdfs:
            st.error("Upload at least one PDF.")
        else:
            with st.spinner("Building index from PDFs..."):
                idx, chunks = build_index_from_pdfs(pdfs)
            st.session_state.index = idx
            st.session_state.chunks = chunks
            st.success(f"Index built. Chunks: {len(chunks)}")

    st.divider()
    st.header("LLM settings")

    provider = st.selectbox(
        "Provider",
        options=["auto", "hf-inference"],
        index=0,
        help="If hf-inference fails or shows Supported tasks: None, use auto.",
        key="provider_sel",
    )

    hf_model = st.selectbox(
        "Model",
        options=[
            "HuggingFaceH4/zephyr-7b-beta",
            "microsoft/phi-2",
            "google/gemma-2b-it",
            "tiiuae/falcon-7b-instruct",
            "mistralai/Mistral-7B-Instruct-v0.3",
        ],
        index=0,
        key="model_sel",
    )

    if st.button("Test LLM"):
        test_q = "Explain greisen alteration in one short paragraph."
        st.write(answer_with_hf(test_q, evidence=[], model=hf_model, provider=provider))


# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs(["RAG QA", "Benchmark (GeoGPT)", "SFT Export", "Sn–REE Lab"])


# -----------------------------
# TAB 1: RAG QA
# -----------------------------
with tab1:
    st.header("Ask a question (RAG QA)")
    q = st.text_input("Question", placeholder="Example: How does greisen alteration relate to tin mineralization?", key="rag_q")

    if st.button("Retrieve and Answer", key="rag_run"):
        if not q.strip():
            st.warning("Type a question first.")
        elif st.session_state.index is None:
            st.warning("Upload PDFs and click Build PDF Index first (in the sidebar).")
        else:
            hits = retrieve(q, st.session_state.index, st.session_state.chunks, k=5)

            st.subheader("Evidence (from your PDFs)")
            for i, h in enumerate(hits, 1):
                st.markdown(f"**[S{i}] {h['source']} | page {h['page']}**")
                st.write(h["text"])
                st.divider()

            st.subheader("Answer (with citations)")
            with st.spinner("Calling Hugging Face inference..."):
                final = answer_with_hf(q, hits, model=hf_model, provider=provider)
            st.write(final)

    st.info(
        "Real data needed here: upload real geology PDFs. "
        "GeoGPT-CoT-QA is used in the Benchmark and SFT Export tabs."
    )


# -----------------------------
# TAB 2: Benchmark (GeoGPT)
# -----------------------------
with tab2:
    st.header("GeoGPT-CoT-QA Benchmark (Closed-book)")
    st.caption("This tests the model on GeoGPT questions without using your PDFs.")

    df = None
    try:
        df = load_geogpt_csv("data/geogpt_cot_qa.csv")
        st.success(f"Loaded GeoGPT dataset. Rows: {len(df)}")
    except Exception as e:
        st.error(f"Could not load data/geogpt_cot_qa.csv. Error: {type(e).__name__}: {e}")

    n = st.slider("Number of questions", min_value=5, max_value=100, value=20, step=5, key="bench_n")
    seed = st.number_input("Random seed", value=42, step=1, key="bench_seed")

    if df is not None and st.button("Run benchmark", key="bench_run"):
        def closed_book_answer_fn(question: str) -> str:
            return answer_with_hf(question, evidence=[], model=hf_model, provider=provider)

        with st.spinner("Running benchmark..."):
            results = run_closed_book_benchmark(df, closed_book_answer_fn, n=int(n), seed=int(seed))

        st.success("Benchmark complete.")
        st.write(
            {
                "n": int(len(results)),
                "exact_match_avg": float(results["exact_match"].mean()),
                "similarity_avg": float(results["similarity"].mean()),
                "model": hf_model,
                "provider": provider,
            }
        )

        st.dataframe(results, use_container_width=True)

        csv_bytes = results.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download results CSV",
            data=csv_bytes,
            file_name="geogpt_benchmark_results.csv",
            mime="text/csv",
            key="bench_dl",
        )


# -----------------------------
# TAB 3: SFT Export
# -----------------------------
with tab3:
    st.header("SFT Export (GeoGPT to JSONL)")
    st.caption("Creates prompt/response JSONL pairs using question, think, answer.")

    df = None
    try:
        df = load_geogpt_csv("data/geogpt_cot_qa.csv")
        st.success(f"Loaded GeoGPT dataset. Rows: {len(df)}")
        st.dataframe(df[["question", "think", "answer"]].head(3), use_container_width=True)
    except Exception as e:
        st.error(f"Could not load data/geogpt_cot_qa.csv. Error: {type(e).__name__}: {e}")

    if df is not None and st.button("Create JSONL exports", key="sft_make"):
        def make_example(row):
            q2 = str(row["question"])
            think = str(row["think"])
            ans = str(row["answer"])
            prompt = "Question:\n" + q2 + "\n\nAnswer with step-by-step reasoning."
            response = think.strip() + "\n\nFinal Answer: " + ans.strip()
            return {"prompt": prompt, "response": response}

        examples = [make_example(r) for _, r in df.iterrows()]
        train, val = train_test_split(examples, test_size=0.02, random_state=42)

        train_jsonl = "\n".join(json.dumps(x, ensure_ascii=False) for x in train).encode("utf-8")
        val_jsonl = "\n".join(json.dumps(x, ensure_ascii=False) for x in val).encode("utf-8")

        st.success(f"Created JSONL. Train: {len(train)}  Val: {len(val)}")

        st.download_button(
            "Download train JSONL",
            data=train_jsonl,
            file_name="sft_train.jsonl",
            mime="application/jsonl",
            key="sft_train_dl",
        )
        st.download_button(
            "Download val JSONL",
            data=val_jsonl,
            file_name="sft_val.jsonl",
            mime="application/jsonl",
            key="sft_val_dl",
        )


# -----------------------------
# TAB 4: Sn–REE Lab v2 (stable reruns)
# -----------------------------
with tab4:
    st.header("Sn–REE Lab (Jos Plateau Younger Granites)")
    st.caption("XRF + depth intervals + hotspots + ML baselines + geology-style logs + map-click selection (stable reruns).")

    xrf_file = st.file_uploader("Upload XRF CSV", type=["csv"], key="xrf_csv")

    colA, colB, colC = st.columns(3)
    with colA:
        oxide_mode = st.selectbox("Geochem mode", ["auto", "prefer_oxides", "elements_only"], index=0, key="ox_mode")
    with colB:
        hotspot_method = st.selectbox("Hotspot method", ["Gi*", "Moran", "Grid"], index=0, key="hs_method")
    with colC:
        interval_choice = st.selectbox("Depth interval", ["All", "A", "B", "C", "D"], index=0, key="hs_interval")

    colD, colE, colF = st.columns(3)
    with colD:
        hotspot_var = st.selectbox("Hotspot variable", ["SnO2", "Ta2O5", "ZrO2", "FracProxy"], index=0, key="hs_var")
    with colE:
        knn_k = st.slider("k neighbors (Gi*/Moran)", 4, 20, 8, 1, key="hs_k")
    with colF:
        grid_cell = st.slider("Grid cell size (m)", 100, 1000, 250, 50, key="hs_grid")

    st.subheader("Exploration task")
    snree_task = st.text_input(
        "Task",
        value="Assess Sn–REE prospectivity from XRF and depth intervals; map hotspots; propose trench orientation and follow-up exploration vector.",
        key="snree_task_v2",
    )

    # IMPORTANT: compute only on button click, store in session_state
    run = st.button("Run Sn–REE Lab v2", key="snree_run")

    if run:
        if xrf_file is None:
            st.error("Please upload an XRF CSV first.")
        else:
            try:
                xrf_df = load_xrf_csv(xrf_file)

                xrf_df = ensure_oxides(
                    xrf_df,
                    prefer_existing_oxides=(oxide_mode != "elements_only"),
                    allow_element_to_oxide=(oxide_mode != "elements_only"),
                )
                xrf_df = add_depth_interval_fields(xrf_df, pit_depth_col="depth")

                xrf_df["FracProxy"] = (
                    pd.to_numeric(xrf_df.get("Rb2O", 0), errors="coerce").fillna(0.0)
                    + pd.to_numeric(xrf_df.get("Cs2O", 0), errors="coerce").fillna(0.0)
                ) / (pd.to_numeric(xrf_df.get("SrO", 0), errors="coerce").fillna(0.0) + 1e-6)

                xrf_summary = summarize_xrf(xrf_df, oxide_mode=oxide_mode)
                rating, signals = quick_prospectivity(xrf_summary)

                dd = xrf_df.copy()
                if interval_choice != "All":
                    dd = dd[dd["INTERVAL"].astype(str).str.upper().str.strip() == interval_choice].copy()

                hot_df, hot_meta = None, {}
                if "lat" in dd.columns and "lon" in dd.columns and len(dd.dropna(subset=["lat", "lon"])) > 0:
                    hot_df, hot_meta = hotspot_analysis(
                        dd.dropna(subset=["lat", "lon"]),
                        value_col=hotspot_var,
                        method=hotspot_method,
                        k=int(knn_k),
                        grid_cell_m=float(grid_cell),
                        z_thresh=1.0,
                    )

                evidence = []
                if st.session_state.index is not None:
                    evidence = retrieve(snree_task, st.session_state.index, st.session_state.chunks, k=6)

                ml_bundle = None
                try:
                    feat_df = build_features(xrf_df)
                    sn = pd.to_numeric(feat_df.get("SnO2", np.nan), errors="coerce")
                    q1, q2 = sn.quantile(0.33), sn.quantile(0.66)

                    def sn_label(v):
                        if pd.isna(v):
                            return np.nan
                        if v <= q1:
                            return "Low"
                        if v <= q2:
                            return "Medium"
                        return "High"

                    feat_df["prospectivity_class"] = sn.apply(sn_label)
                    ml_bundle = train_baselines(feat_df, target_col="prospectivity_class", test_size=0.25, seed=42)
                except Exception:
                    ml_bundle = None

                preds = None
                if "hole_id" in xrf_df.columns:
                    preds = within_hole_prediction(xrf_df, hole_col="hole_id", value_col=hotspot_var)

                st.session_state.snree.update(
                    {
                        "xrf_df": xrf_df,
                        "xrf_summary": xrf_summary,
                        "rating": rating,
                        "signals": signals,
                        "hot_df": hot_df,
                        "hot_meta": hot_meta,
                        "evidence": evidence,
                        "ml_bundle": ml_bundle,
                        "preds": preds,
                        "last_run_ok": True,
                        "last_run_params": {
                            "oxide_mode": oxide_mode,
                            "hotspot_method": hotspot_method,
                            "interval_choice": interval_choice,
                            "hotspot_var": hotspot_var,
                            "knn_k": int(knn_k),
                            "grid_cell": float(grid_cell),
                        },
                    }
                )

                st.success("Sn–REE Lab computed and saved. You can interact with the map and plots without losing outputs.")
            except Exception as e:
                st.session_state.snree["last_run_ok"] = False
                st.error(f"Sn–REE Lab failed: {type(e).__name__}: {e}")

    # Render from stored results (survives reruns)
    if not st.session_state.snree.get("last_run_ok"):
        st.info("Upload an XRF CSV and click **Run Sn–REE Lab v2** to generate outputs.")
        st.stop()

    xrf_df = st.session_state.snree["xrf_df"]
    xrf_summary = st.session_state.snree["xrf_summary"]
    rating = st.session_state.snree["rating"]
    signals = st.session_state.snree["signals"]
    hot_df = st.session_state.snree["hot_df"]
    hot_meta = st.session_state.snree["hot_meta"]
    evidence = st.session_state.snree["evidence"] or []
    ml_bundle = st.session_state.snree["ml_bundle"]
    preds = st.session_state.snree["preds"]

    st.subheader("Rules-only screening")
    st.write({"prospectivity": rating, "signals": signals})
    if xrf_summary and xrf_summary.get("FracProxy_definition"):
        st.caption(xrf_summary.get("FracProxy_definition"))
    if xrf_summary and xrf_summary.get("FracProxy_rationale"):
        st.caption(xrf_summary.get("FracProxy_rationale"))

    st.subheader("Evidence used (PDF snippets)")
    if evidence:
        for i, h in enumerate(evidence, 1):
            st.markdown(f"**[S{i}] {h['source']} | page {h['page']}**")
            st.write(h["text"])
            st.divider()
    else:
        st.info("No PDF evidence used. Build the PDF index in the sidebar to enable citations.")

    st.subheader("Hotspot analysis (table + orientation)")
    if hot_df is None:
        st.info("Hotspot analysis not available (missing lat/lon or too few points).")
    else:
        cols = ["hole_id", "INTERVAL", "lat", "lon", "cluster", "method"]
        vcol = st.session_state.snree.get("last_run_params", {}).get("hotspot_var", None)
        if vcol and vcol in hot_df.columns:
            cols.insert(4, vcol)
        cols = [c for c in cols if c in hot_df.columns]
        st.dataframe(hot_df[cols].head(300), use_container_width=True)

        orient = (hot_meta or {}).get("orientation", {})
        if orient.get("trend_azimuth_deg") is not None:
            st.success(
                f"Inferred mineralization trend azimuth: {orient['trend_azimuth_deg']:.1f}° | "
                f"Recommended trench azimuth (perpendicular): {orient['trench_azimuth_deg']:.1f}° "
                f"(based on {orient.get('n_high', 0)} high-cluster points)"
            )
        else:
            st.info("Not enough high-cluster points to infer a stable trend. Add more samples or use Grid binning.")

    st.subheader("Hotspot map (click a point to open its log)")
    if hot_df is None:
        st.info("Run hotspots with valid lat/lon to enable the clickable map.")
    else:
        map_df = hot_df.dropna(subset=["lat", "lon", "hole_id"]).copy()
        map_df["lat"] = pd.to_numeric(map_df["lat"], errors="coerce")
        map_df["lon"] = pd.to_numeric(map_df["lon"], errors="coerce")
        map_df = map_df.dropna(subset=["lat", "lon"])

        if len(map_df) == 0:
            st.info("No mappable hotspot points (missing lat/lon).")
        else:
            vcol = st.session_state.snree.get("last_run_params", {}).get("hotspot_var", None)

            center = [float(map_df["lat"].mean()), float(map_df["lon"].mean())]
            m = folium.Map(location=center, zoom_start=12, control_scale=True, tiles="OpenStreetMap")

            def color_for_cluster(c):
                c = str(c).upper()
                if "HH" in c or "HOT" in c or "HIGH" in c:
                    return "red"
                if "LL" in c or "COLD" in c or "LOW" in c:
                    return "blue"
                if "MED" in c:
                    return "orange"
                return "gray"

            for _, r in map_df.iterrows():
                hole = str(r["hole_id"])
                iv = str(r.get("INTERVAL", ""))
                cl = str(r.get("cluster", ""))
                val = r.get(vcol, None) if vcol else None
                popup = f"hole_id: {hole}<br>Interval: {iv}<br>Cluster: {cl}"
                if vcol:
                    popup += f"<br>{vcol}: {val}"
                folium.CircleMarker(
                    location=[float(r["lat"]), float(r["lon"])],
                    radius=6,
                    color=color_for_cluster(cl),
                    fill=True,
                    fill_opacity=0.8,
                    popup=folium.Popup(popup, max_width=300),
                ).add_to(m)

            st.caption("Click a point (or near it). The app snaps to the nearest point and selects that hole.")
            map_state = st_folium(m, height=450, width=None, key="snree_map")

            clicked = map_state.get("last_clicked", None)
            if clicked and "lat" in clicked and "lng" in clicked:
                click_lat = float(clicked["lat"])
                click_lon = float(clicked["lng"])
                row, dist_m = nearest_row_by_click(map_df, click_lat, click_lon, lat_col="lat", lon_col="lon")
                new_hole = str(row["hole_id"])
                if new_hole != st.session_state.selected_hole_from_map:
                    st.session_state.selected_hole_from_map = new_hole
                    st.success(f"Selected hole_id: {new_hole} (snapped {dist_m:.0f} m)")

    st.subheader("ML baselines (experimental)")
    if ml_bundle is None:
        st.info("ML baselines not available (dataset too small/imbalanced or missing features).")
    else:
        st.write({"n_train": ml_bundle["n_train"], "n_test": ml_bundle["n_test"], "classes": ml_bundle["classes"]})
        for name, info in ml_bundle["models"].items():
            st.markdown(f"### Model: {name}")
            st.write({"accuracy": info.get("accuracy"), "features": info.get("features")})
            if "feature_importance" in info:
                st.write({"feature_importance": info["feature_importance"]})

    st.subheader("Within-hole depth profile + strip log (GIS/geology view)")
    st.caption("Left: measured points + predicted trend. Right: interval strip log with your lithology/alteration notes.")

    profile_var = st.selectbox("Profile variable", ["SnO2", "Ta2O5", "FracProxy", "ZrO2"], index=0, key="prof_var")
    show_inferred_hints = st.checkbox("Show quick inferred hints on strip log (experimental)", value=True, key="hint_toggle")

    with st.expander("Strip log notes (optional, user input)", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Lithology notes**")
            st.session_state.striplog_notes["A"]["lithology"] = st.text_input(
                "A (0–0.3 m)", st.session_state.striplog_notes["A"]["lithology"], key="lith_A"
            )
            st.session_state.striplog_notes["B"]["lithology"] = st.text_input(
                "B (0.3–1.0 m)", st.session_state.striplog_notes["B"]["lithology"], key="lith_B"
            )
            st.session_state.striplog_notes["C"]["lithology"] = st.text_input(
                "C (1.0–2.0 m)", st.session_state.striplog_notes["C"]["lithology"], key="lith_C"
            )
            st.session_state.striplog_notes["D"]["lithology"] = st.text_input(
                "D (>2.0 m)", st.session_state.striplog_notes["D"]["lithology"], key="lith_D"
            )
        with c2:
            st.markdown("**Alteration notes**")
            st.session_state.striplog_notes["A"]["alteration"] = st.text_input(
                "A alteration", st.session_state.striplog_notes["A"]["alteration"], key="alt_A"
            )
            st.session_state.striplog_notes["B"]["alteration"] = st.text_input(
                "B alteration", st.session_state.striplog_notes["B"]["alteration"], key="alt_B"
            )
            st.session_state.striplog_notes["C"]["alteration"] = st.text_input(
                "C alteration", st.session_state.striplog_notes["C"]["alteration"], key="alt_C"
            )
            st.session_state.striplog_notes["D"]["alteration"] = st.text_input(
                "D alteration", st.session_state.striplog_notes["D"]["alteration"], key="alt_D"
            )

    if preds is None or not preds:
        st.info("Within-hole prediction not available (need hole_id and at least 3 samples per hole).")
    else:
        hole_list = sorted(list(preds.keys()))
        default_hole = st.session_state.get("selected_hole_from_map", None)
        default_index = hole_list.index(default_hole) if default_hole in hole_list else 0

        selected_hole = st.selectbox("Select hole", options=hole_list, index=default_index, key="hole_sel")
        p = preds[selected_hole]

        meas_tbl = pd.DataFrame(
            {
                "depth_mid_m": p["measured_depth_m"],
                "value": p["measured_value"],
                "INTERVAL": p.get("measured_interval", [""] * len(p["measured_depth_m"])),
            }
        ).sort_values("depth_mid_m")

        def norm_interval(s: str) -> str:
            s2 = str(s).strip().upper()
            return s2 if s2 in ["A", "B", "C", "D"] else "NA"

        meas_tbl["INTERVAL_N"] = meas_tbl["INTERVAL"].apply(norm_interval)

        maxd = float(p["max_depth_m"])
        bands = [
            ("A", 0.0, min(0.3, maxd)),
            ("B", 0.3, min(1.0, maxd)),
            ("C", 1.0, min(2.0, maxd)),
            ("D", 2.0, maxd),
        ]

        hints = {"A": "", "B": "", "C": "", "D": ""}
        if show_inferred_hints:
            overall_med = float(pd.to_numeric(meas_tbl["value"], errors="coerce").median())
            for iv in ["A", "B", "C", "D"]:
                g = meas_tbl[meas_tbl["INTERVAL_N"] == iv]
                if len(g) >= 1:
                    iv_med = float(pd.to_numeric(g["value"], errors="coerce").median())
                    hints[iv] = "Relative enrichment" if iv_med > overall_med else "Background to moderate"

        fig = plt.figure(figsize=(10, 6))
        gs = gridspec.GridSpec(1, 2, width_ratios=[3.4, 1.6], wspace=0.25)
        ax = fig.add_subplot(gs[0, 0])
        axlog = fig.add_subplot(gs[0, 1])

        for name, y0, y1 in bands:
            if y1 > y0:
                ax.axhspan(y0, y1, alpha=0.08)
                axlog.axhspan(y0, y1, alpha=0.08)

        ax.plot(p["predicted"], p["depth_grid_m"], label="Predicted trend", linewidth=2)

        for iv in ["A", "B", "C", "D"]:
            g = meas_tbl[meas_tbl["INTERVAL_N"] == iv]
            if len(g):
                ax.scatter(g["value"], g["depth_mid_m"], s=60, label=f"Measured {iv}")

        g0 = meas_tbl[meas_tbl["INTERVAL_N"] == "NA"]
        if len(g0):
            ax.scatter(g0["value"], g0["depth_mid_m"], s=60, label="Measured (unlabeled)")

        ax.set_xlabel(f"{profile_var} (units as in CSV)")
        ax.set_ylabel("Depth (m)")
        ax.invert_yaxis()
        ax.grid(True)
        ax.legend(loc="best")

        axlog.set_xlim(0, 1)
        axlog.set_xticks([])
        axlog.set_xlabel("Strip log")
        axlog.set_ylabel("Depth (m)")
        axlog.invert_yaxis()
        axlog.grid(False)

        for name, y0, y1 in bands:
            if y1 <= y0:
                continue
            ymid = (y0 + y1) / 2.0
            notes = st.session_state.striplog_notes.get(name, {"lithology": "", "alteration": ""})
            lith = notes.get("lithology", "").strip()
            alt = notes.get("alteration", "").strip()
            hint = hints.get(name, "").strip()

            if name != "D":
                header = f"{name} ({y0:.1f}–{y1:.1f} m)"
            else:
                header = "D (>2.0 m)"

            lines = [header]
            if lith:
                lines.append(f"Lith: {lith}")
            if alt:
                lines.append(f"Alt: {alt}")
            if hint:
                lines.append(f"Hint: {hint}")

            axlog.text(0.02, ymid, "\n".join(lines), va="center", fontsize=9)

        st.pyplot(fig, clear_figure=True)

        st.dataframe(
            meas_tbl[["depth_mid_m", "INTERVAL", "value"]].rename(columns={"value": f"{profile_var}_measured"}),
            use_container_width=True,
        )

    with st.expander("Generate geology narrative (optional)", expanded=False):
        prompt = build_snree_prompt(
            question=snree_task,
            xrf_summary=xrf_summary,
            evidence=evidence,
            prospectivity=rating,
            signals=signals,
        )
        st.text_area("Prompt preview", value=prompt, height=240, key="prompt_preview")

        if st.button("Generate narrative with LLM", key="gen_narr"):
            with st.spinner("Generating..."):
                out = answer_with_hf(prompt, evidence=[], model=hf_model, provider=provider)
            st.write(out)
