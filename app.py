import os
import json
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import matplotlib.pyplot as plt
from core.chemistry import ensure_oxides
from core.snree_hotspots import hotspot_analysis
from core.snree_ml import build_features, train_baselines, within_hole_prediction

import pydeck as pdk
import pandas as pd


from sklearn.model_selection import train_test_split

from core.rag import build_index_from_pdfs, retrieve
from core.llm_hf import answer_with_hf, generate_from_prompt
from core.geogpt import load_geogpt_csv
from core.benchmark import run_closed_book_benchmark

from core.snree_lab import (
    load_xrf_csv,
    add_depth_interval_fields,
    summarize_xrf,
    quick_prospectivity,
    build_snree_prompt,
)

from core.snree_ml import (
    build_features,
    train_baselines,
    within_hole_prediction,
)

st.set_page_config(page_title="GeoMinerAI", layout="wide")

st.title("GeoMinerAI")
st.caption("Upload geoscience PDFs, retrieve evidence, and answer with citations using a Hugging Face model.")

# Keep index and chunks in memory between button clicks
if "index" not in st.session_state:
    st.session_state.index = None
    st.session_state.chunks = None

# Keep ML bundle between runs
if "snree_ml_bundle" not in st.session_state:
    st.session_state.snree_ml_bundle = None

# Sidebar stays global (it affects all tabs)
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
        help="If hf-inference fails or shows Supported tasks: None, use auto."
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

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["RAG QA", "Benchmark (GeoGPT)", "SFT Export", "Sn–REE Lab"])

# -------------------------
# TAB 1: RAG QA
# -------------------------
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

# -------------------------
# TAB 2: Benchmark (GeoGPT)
# -------------------------
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
        st.write({
            "n": int(len(results)),
            "exact_match_avg": float(results["exact_match"].mean()),
            "similarity_avg": float(results["similarity"].mean()),
            "model": hf_model,
            "provider": provider,
        })

        st.dataframe(results, use_container_width=True)

        csv_bytes = results.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download results CSV",
            data=csv_bytes,
            file_name="geogpt_benchmark_results.csv",
            mime="text/csv",
        )

# -------------------------
# TAB 3: SFT Export
# -------------------------
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

# -------------------------
# TAB 4: Sn–REE Lab v1 + v2
# -------------------------
with tab4:
    st.header("Sn–REE Lab (Jos Plateau Younger Granites)")
    st.caption("XRF + depth intervals + hotspot mapping + ML baselines + orientation-guided trench suggestions.")

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
        key="snree_task_v2"
    )

    if st.button("Run Sn–REE Lab v2"):
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
            xrf_df["FracProxy"] = (pd.to_numeric(xrf_df.get("Rb2O", 0), errors="coerce").fillna(0.0)
                                   + pd.to_numeric(xrf_df.get("Cs2O", 0), errors="coerce").fillna(0.0)) / (
                                      pd.to_numeric(xrf_df.get("SrO", 0), errors="coerce").fillna(0.0) + 1e-6
                                   )

            # Summaries and screening
            xrf_summary = summarize_xrf(xrf_df, oxide_mode=oxide_mode)
            rating, signals = quick_prospectivity(xrf_summary)

            st.subheader("Rules-only screening")
            st.write({"prospectivity": rating, "signals": signals})
            st.caption(xrf_summary.get("FracProxy_definition"))
            st.caption(xrf_summary.get("FracProxy_rationale"))

            # Interval filter
            dd = xrf_df.copy()
            if interval_choice != "All":
                dd = dd[dd["INTERVAL"].astype(str).str.upper().str.strip() == interval_choice].copy()

            # Hotspots
            st.subheader("Hotspot analysis")
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

                # Map using pydeck with cluster-based colors
                def color_for_cluster(c: str):
                    c = str(c)
                    if c in ["Hot", "HH", "High"]:
                        return [200, 30, 30, 160]
                    if c in ["Cold", "LL", "Low"]:
                        return [30, 30, 200, 160]
                    if c in ["Medium"]:
                        return [240, 180, 30, 160]
                    if c in ["Outlier"]:
                        return [150, 30, 150, 160]
                    return [90, 90, 90, 120]

                hot_df["_color"] = hot_df["cluster"].apply(color_for_cluster)

                layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=hot_df,
                    get_position=["lon", "lat"],
                    get_fill_color="_color",
                    get_radius=35,
                    pickable=True,
                )
                view = pdk.ViewState(
                    latitude=float(hot_df["lat"].mean()),
                    longitude=float(hot_df["lon"].mean()),
                    zoom=11,
                )
                deck = pdk.Deck(layers=[layer], initial_view_state=view, tooltip={"text": "{cluster}\n" + hotspot_var + ": {" + hotspot_var + "}"})
                st.pydeck_chart(deck)

                # Show labeled table for decisions
                cols = ["hole_id", "INTERVAL", "lat", "lon", hotspot_var, "cluster", "method"]
                cols = [c for c in cols if c in hot_df.columns]
                st.dataframe(hot_df[cols].head(200), use_container_width=True)


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

            # Build evidence if PDFs are indexed
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

            # ML baselines
            st.subheader("ML baselines (experimental)")
            feat_df = build_features(xrf_df)

            # Choose a label source for training, for now use rules-only label per row as baseline
            # Later you can switch to field labels when you have them
            # Simple per-row rule label: compare to global quantiles
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

            bundle = train_baselines(feat_df, target_col="prospectivity_class", test_size=0.25, seed=42)
            st.write({"n_train": bundle["n_train"], "n_test": bundle["n_test"], "classes": bundle["classes"]})

            for name, info in bundle["models"].items():
                st.markdown(f"### Model: {name}")
                st.write({"accuracy": info["accuracy"], "features": info["features"]})
                if "feature_importance" in info:
                    st.write({"feature_importance": info["feature_importance"]})

            # Within-hole prediction
            st.subheader("Within-hole depth profile (measured points + predicted curve)")
            st.caption("Depth increases downward. Shaded bands show A/B/C/D intervals.")

            # Let user choose what variable to profile
            profile_var = st.selectbox("Profile variable", ["SnO2", "Ta2O5", "FracProxy", "ZrO2"], index=0)

            if "hole_id" in xrf_df.columns:
                preds = within_hole_prediction(xrf_df, hole_col="hole_id", value_col=profile_var)

                if not preds:
                    st.info("Not enough points per hole to fit a depth trend (need at least 3 interval samples in a hole).")
                else:
                    hole_list = sorted(list(preds.keys()))
                    selected_hole = st.selectbox("Select hole", options=hole_list, index=0)

                    p = preds[selected_hole]

                    # Build measured table for plotting
                    meas_tbl = pd.DataFrame({
                        "depth_mid_m": p["measured_depth_m"],
                        "value": p["measured_value"],
                        "INTERVAL": p.get("measured_interval", [""] * len(p["measured_depth_m"])),
                    }).sort_values("depth_mid_m")

                    # Interval colors (consistent, geology-friendly)
                    interval_colors = {
                        "A": "tab:green",
                        "B": "tab:orange",
                        "C": "tab:blue",
                        "D": "tab:red",
                    }

                    def norm_interval(s: str) -> str:
                        s2 = str(s).strip().upper()
                        return s2 if s2 in interval_colors else "NA"

                    meas_tbl["INTERVAL_N"] = meas_tbl["INTERVAL"].apply(norm_interval)

                    fig = plt.figure()
                    ax = fig.add_subplot(111)

                    # Optional upgrade: shaded depth bands for A/B/C/D
                    # A: 0-0.3, B: 0.3-1.0, C: 1.0-2.0, D: 2.0-max_depth
                    maxd = float(p["max_depth_m"])
                    bands = [
                        ("A", 0.0, min(0.3, maxd)),
                        ("B", 0.3, min(1.0, maxd)),
                        ("C", 1.0, min(2.0, maxd)),
                        ("D", 2.0, maxd),
                    ]

                    for name, y0, y1 in bands:
                        if y1 > y0:
                            ax.axhspan(y0, y1, alpha=0.08)
                            ax.text(
                                x=0.01, y=(y0 + y1) / 2.0,
                                s=name,
                                transform=ax.get_yaxis_transform(),
                                va="center",
                                fontsize=9
                            )

                    # Predicted curve
                    ax.plot(p["predicted"], p["depth_grid_m"], label="Predicted trend", linewidth=2)

                    # Measured points colored by interval
                    for iv, g in meas_tbl.groupby("INTERVAL_N"):
                        if iv in interval_colors:
                            ax.scatter(g["value"], g["depth_mid_m"], label=f"Measured {iv}", s=55)
                        else:
                            ax.scatter(g["value"], g["depth_mid_m"], label="Measured (unlabeled)", s=55)

                    ax.set_xlabel(f"{profile_var} (same units as your CSV)")
                    ax.set_ylabel("Depth (m)")
                    ax.invert_yaxis()
                    ax.grid(True)
                    ax.legend(loc="best")

                    st.pyplot(fig, clear_figure=True)

                    # Clean table (no JSON)
                    st.dataframe(
                        meas_tbl[["depth_mid_m", "INTERVAL", "value"]].rename(columns={"value": f"{profile_var}_measured"}),
                        use_container_width=True
                    )

            else:
                st.info("No hole_id column found. Add hole_id to enable within-hole profiling.")

