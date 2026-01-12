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

if "snree_ml_bundle" not in st.session_state:
    st.session_state.snree_ml_bundle = None

if "selected_hole_from_map" not in st.session_state:
    st.session_state.selected_hole_from_map = None

if "striplog_notes" not in st.session_state:
    st.session_state.striplog_notes = {
        "A": {"lithology": "", "alteration": ""},
        "B": {"lithology": "", "alteration": ""},
        "C": {"lithology": "", "alteration": ""},
        "D": {"lithology": "", "alteration": ""},
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
    q = st.text_input("Question", placeholder="Example: How does greisen alteration relate to tin mineralization?")

    if st.button("Retrieve and Answer"):
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

    n = st.slider("Number of questions", min_value=5, max_value=100, value=20, step=5)
    seed = st.number_input("Random seed", value=42, step=1)

    if df is not None and st.button("Run benchmark"):
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

    if df is not None and st.button("Create JSONL exports"):
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
        )
        st.download_button(
            "Download val JSONL",
            data=val_jsonl,
            file_name="sft_val.jsonl",
            mime="application/jsonl",
        )


# -----------------------------
# TAB 4: Sn–REE Lab v2
# -----------------------------
with tab4:
    st.header("Sn–REE Lab (Jos Plateau Younger Granites)")
    st.caption("XRF + depth intervals + hotspot mapping + ML baselines + geology-style logs + map-click selection.")

    xrf_file = st.file_uploader("Upload XRF CSV", type=["csv"], key="xrf_csv")

    colA, colB, colC = st.columns(3)
    with colA:
        oxide_mode = st.selectbox("Geochem mode", ["auto", "prefer_oxides", "elements_only"], index=0)
    with colB:
        hotspot_method = st.selectbox("Hotspot method", ["Gi*", "Moran", "Grid"], index=0)
    with colC:
        interval_choice = st.selectbox("Depth interval", ["All", "A", "B", "C", "D"], index=0)

    colD, colE, colF = st.columns(3)
    with colD:
        hotspot_var = st.selectbox("Hotspot variable", ["SnO2", "Ta2O5", "ZrO2", "FracProxy"], index=0)
    with colE:
        knn_k = st.slider("k neighbors (Gi*/Moran)", 4, 20, 8, 1)
    with colF:
        grid_cell = st.slider("Grid cell size (m)", 100, 1000, 250, 50)

    st.subheader("Exploration task")
    snree_task = st.text_input(
        "Task",
        value="Assess Sn–REE prospectivity from XRF and depth intervals; map hotspots; propose trench orientation and follow-up exploration vector.",
        key="snree_task_v2",
    )

    if st.button("Run Sn–REE Lab"):
        if xrf_file is None:
            st.error("Please upload an XRF CSV first.")
        else:
            xrf_df = load_xrf_csv(xrf_file)

            # Ensure oxides and compute FracProxy
            xrf_df = ensure_oxides(
                xrf_df,
                prefer_existing_oxides=(oxide_mode != "elements_only"),
                allow_element_to_oxide=(oxide_mode != "elements_only"),
            )
            xrf_df = add_depth_interval_fields(xrf_df, pit_depth_col="depth")

            # Compute FracProxy explicitly
            xrf_df["FracProxy"] = (
                pd.to_numeric(xrf_df.get("Rb2O", 0), errors="coerce").fillna(0.0)
                + pd.to_numeric(xrf_df.get("Cs2O", 0), errors="coerce").fillna(0.0)
            ) / (pd.to_numeric(xrf_df.get("SrO", 0), errors="coerce").fillna(0.0) + 1e-6)

            # Summaries and screening
            xrf_summary = summarize_xrf(xrf_df, oxide_mode=oxide_mode)
            rating, signals = quick_prospectivity(xrf_summary)

            st.subheader("Rules-only screening")
            st.write({"prospectivity": rating, "signals": signals})
            if xrf_summary.get("FracProxy_definition"):
                st.caption(xrf_summary.get("FracProxy_definition"))
            if xrf_summary.get("FracProxy_rationale"):
                st.caption(xrf_summary.get("FracProxy_rationale"))

            # Interval filter for hotspot display
            dd = xrf_df.copy()
            if interval_choice != "All":
                dd = dd[dd["INTERVAL"].astype(str).str.upper().str.strip() == interval_choice].copy()

            # -----------------------------
            # Hotspot analysis
            # -----------------------------
            st.subheader("Hotspot analysis")
            hot_df = None
            meta = {}

            if "lat" not in dd.columns or "lon" not in dd.columns:
                st.warning("No lat/lon columns found. Hotspot analysis requires lat and lon.")
            else:
                hot_df, meta = hotspot_analysis(
                    dd.dropna(subset=["lat", "lon"]),
                    value_col=hotspot_var,
                    method=hotspot_method,
                    k=int(knn_k),
                    grid_cell_m=float(grid_cell),
                    z_thresh=1.0,
                )
                st.write(meta)

                cols = ["hole_id", "INTERVAL", "lat", "lon", hotspot_var, "cluster", "method"]
                cols = [c for c in cols if c in hot_df.columns]
                st.dataframe(hot_df[cols].head(300), use_container_width=True)

                # Orientation and trench azimuth
                orient = meta.get("orientation", {})
                if orient.get("trend_azimuth_deg") is not None:
                    st.success(
                        f"Inferred mineralization trend azimuth: {orient['trend_azimuth_deg']:.1f}° | "
                        f"Recommended trench azimuth (perpendicular): {orient['trench_azimuth_deg']:.1f}° "
                        f"(based on {orient.get('n_high', 0)} high-cluster points)"
                    )
                else:
                    st.info("Not enough high-cluster points to infer a stable trend. Add more samples or use Grid binning.")

            # -----------------------------
            # Clickable hotspot map -> select hole_id
            # -----------------------------
            st.subheader("Hotspot map (click a point to open its log)")

            if hot_df is None:
                st.info("Run hotspot analysis first to enable the map.")
            else:
                map_df = hot_df.dropna(subset=["lat", "lon", "hole_id"]).copy()
                map_df["lat"] = pd.to_numeric(map_df["lat"], errors="coerce")
                map_df["lon"] = pd.to_numeric(map_df["lon"], errors="coerce")
                map_df = map_df.dropna(subset=["lat", "lon"])

                if len(map_df) == 0:
                    st.info("No mappable hotspot points (missing lat/lon).")
                else:
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
                        val = r.get(hotspot_var, None)
                        popup = f"hole_id: {hole}<br>Interval: {iv}<br>Cluster: {cl}<br>{hotspot_var}: {val}"
                        folium.CircleMarker(
                            location=[float(r["lat"]), float(r["lon"])],
                            radius=6,
                            color=color_for_cluster(cl),
                            fill=True,
                            fill_opacity=0.8,
                            popup=folium.Popup(popup, max_width=300),
                        ).add_to(m)

                    st.caption("Click a point (or near it). The app snaps to the nearest point and selects that hole.")
                    map_state = st_folium(m, height=450, width=None)

                    clicked = map_state.get("last_clicked", None)
                    if clicked and "lat" in clicked and "lng" in clicked:
                        click_lat = float(clicked["lat"])
                        click_lon = float(clicked["lng"])
                        row, dist_m = nearest_row_by_click(map_df, click_lat, click_lon, lat_col="lat", lon_col="lon")
                        st.session_state.selected_hole_from_map = str(row["hole_id"])
                        st.success(
                            f"Selected hole_id from map: {st.session_state.selected_hole_from_map} "
                            f"(snapped {dist_m:.0f} m to nearest point)"
                        )

            # -----------------------------
            # Evidence used (PDF snippets)
            # -----------------------------
            evidence = []
            if st.session_state.index is not None:
                evidence = retrieve(snree_task, st.session_state.index, st.session_state.chunks, k=6)

            st.subheader("Evidence used (PDF snippets)")
            if evidence:
                for i, h in enumerate(evidence, 1):
                    st.markdown(f"**[S{i}] {h['source']} | page {h['page']}**")
                    st.write(h["text"])
                    st.divider()
            else:
                st.info("No PDF evidence used. Build the PDF index in the sidebar to enable citations.")

            # -----------------------------
            # ML baselines (experimental)
            # -----------------------------
            st.subheader("ML baselines (experimental)")
            feat_df = build_features(xrf_df)

            sn = pd.to_numeric(feat_df.get("SnO2", np.nan), errors="coerce")
            q1 = sn.quantile(0.33)
            q2 = sn.quantile(0.66)

            def sn_label(v):
                if pd.isna(v):
                    return np.nan
                if v <= q1:
                    return "Low"
                if v <= q2:
                    return "Medium"
                return "High"

            feat_df["prospectivity_class"] = sn.apply(sn_label)

            try:
                bundle = train_baselines(feat_df, target_col="prospectivity_class", test_size=0.25, seed=42)
                st.session_state.snree_ml_bundle = bundle
                st.write({"n_train": bundle["n_train"], "n_test": bundle["n_test"], "classes": bundle["classes"]})

                for name, info in bundle["models"].items():
                    st.markdown(f"### Model: {name}")
                    st.write({"accuracy": info["accuracy"], "features": info["features"]})
                    if "feature_importance" in info:
                        st.write({"feature_importance": info["feature_importance"]})
            except Exception as e:
                st.warning(f"ML training skipped. Reason: {type(e).__name__}: {e}")

            # -----------------------------
            # Within-hole depth profile + strip log
            # -----------------------------
            st.subheader("Within-hole depth profile + strip log (GIS/geology view)")
            st.caption("Left: measured points + predicted trend. Right: interval strip log with your lithology and alteration notes.")

            profile_var = st.selectbox("Profile variable", ["SnO2", "Ta2O5", "FracProxy", "ZrO2"], index=0)
            show_inferred_hints = st.checkbox("Show quick inferred hints on strip log (experimental)", value=True)

            with st.expander("Strip log notes (optional, user input)", expanded=True):
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

            if "hole_id" not in xrf_df.columns:
                st.info("No hole_id column found. Add hole_id to enable within-hole profiling.")
            else:
                preds = within_hole_prediction(xrf_df, hole_col="hole_id", value_col=profile_var)

                if not preds:
                    st.info("Not enough points per hole to fit a depth trend (need at least 3 interval samples in a hole).")
                else:
                    hole_list = sorted(list(preds.keys()))

                    default_hole = st.session_state.get("selected_hole_from_map", None)
                    default_index = hole_list.index(default_hole) if default_hole in hole_list else 0

                    selected_hole = st.selectbox("Select hole", options=hole_list, index=default_index)
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
                                if iv_med > overall_med:
                                    hints[iv] = "Relative enrichment"
                                else:
                                    hints[iv] = "Background to moderate"

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

            # -----------------------------
            # Optional: build a Sn–REE prompt for your LLM (kept off by default)
            # -----------------------------
            with st.expander("Generate geology narrative (optional)", expanded=False):
                prompt = build_snree_prompt(
                    question=snree_task,
                    xrf_summary=xrf_summary,
                    evidence=evidence,
                    prospectivity=rating,
                    signals=signals,
                )
                st.text_area("Prompt preview", value=prompt, height=250)

                if st.button("Generate narrative with LLM"):
                    with st.spinner("Generating..."):
                        out = answer_with_hf(prompt, evidence=[], model=hf_model, provider=provider)
                    st.write(out)
