import os
import json
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from core.rag import build_index_from_pdfs, retrieve
from core.llm_hf import answer_with_hf, generate_from_prompt
from core.geogpt import load_geogpt_csv
from core.benchmark import run_closed_book_benchmark

from core.snree_lab import (
    load_xrf_csv,
    summarize_xrf,
    quick_prospectivity,
    build_snree_prompt,
)

from core.snree_ml import (
    build_feature_table,
    label_by_rules,
    train_logreg,
    train_xgboost,
    predict_ml,
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
    st.caption("Upload XRF CSV and optionally use your PDF evidence base to generate an exploration vector, targets, maps, profiles, and ML baselines.")

    xrf_file = st.file_uploader("Upload XRF CSV", type=["csv"], key="xrf_csv")

    st.subheader("Sn–REE Assessment Question")
    snree_question = st.text_input(
        "Task",
        value="Assess Sn–REE prospectivity from XRF results and summarize an exploration vector for follow-up work.",
        key="snree_task"
    )

    run_assessment = st.button("Run Sn–REE Assessment (v1)")

    if run_assessment:
        if xrf_file is None:
            st.error("Please upload an XRF CSV first.")
        else:
            # Load and summarize XRF
            xrf_df = load_xrf_csv(xrf_file)
            xrf_summary = summarize_xrf(xrf_df)
            rating, signals = quick_prospectivity(xrf_summary)

            st.subheader("Screening Result (from XRF only)")
            st.write({"prospectivity": rating, "signals": signals})
            st.caption("This is a screening output. Final interpretation below uses PDFs if indexed.")

            # Get PDF evidence if available
            evidence = []
            if st.session_state.index is not None:
                evidence = retrieve(snree_question, st.session_state.index, st.session_state.chunks, k=6)

            st.subheader("Evidence used (PDF snippets)")
            if evidence:
                for i, h in enumerate(evidence, 1):
                    st.markdown(f"**[S{i}] {h['source']} | page {h['page']}**")
                    st.write(h["text"])
                    st.divider()
            else:
                st.info("No PDF index found or no evidence retrieved. Build the PDF index in the sidebar if you want cited geological context.")

            # Build Sn–REE prompt and call model
            prompt = build_snree_prompt(
                question=snree_question,
                xrf_summary=xrf_summary,
                evidence=evidence,
                prospectivity=rating,
                signals=signals,
            )

            st.subheader("Sn–REE Lab Output (LLM)")
            with st.spinner("Generating Sn–REE interpretation..."):
                out = generate_from_prompt(prompt, model=hf_model, provider=provider)
            st.write(out)

            st.subheader("XRF Summary Preview")
            st.json(xrf_summary)

    st.divider()
    st.subheader("Sn–REE Lab v2: Targets, Maps, Depth Profiles, and ML Baselines")

    if xrf_file is None:
        st.info("Upload the XRF CSV above to enable Sn–REE Lab v2.")
    else:
        # Reload fresh for v2 block
        xrf_df = load_xrf_csv(xrf_file)

        feat_df, feat_cols = build_feature_table(xrf_df)
        labeled_df, medians = label_by_rules(feat_df)

        st.write("Detected columns:", feat_cols)
        st.write("Label medians (used for fold):", medians)

        # 1) Labeled training table
        st.subheader("1) Labeled training table (features -> prospectivity_class)")
        st.dataframe(labeled_df.head(25), use_container_width=True)

        csv_labeled = labeled_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download labeled training table CSV",
            data=csv_labeled,
            file_name="snree_labeled_training_table.csv",
            mime="text/csv",
        )

        # 2) Map hotspots
        st.subheader("2) Hotspot map (lat/lon)")
        if "lat" in labeled_df.columns and "lon" in labeled_df.columns and labeled_df["lat"].notna().any() and labeled_df["lon"].notna().any():
            map_df = labeled_df.dropna(subset=["lat", "lon"]).copy()
            st.map(map_df.rename(columns={"lat": "latitude", "lon": "longitude"})[["latitude", "longitude"]])
            st.caption("This is a basic point map. If you want class colors and legend, we can add PyDeck next.")
        else:
            st.info("No usable lat/lon detected for mapping. Ensure columns exist (lat, lon).")

        # 3) Depth profiles per hole_id
        st.subheader("3) Depth profiles per hole_id")
        can_profile = ("hole_id" in labeled_df.columns) and ("depth" in labeled_df.columns) and labeled_df["depth"].notna().any()

        if can_profile:
            hole_ids = sorted(labeled_df["hole_id"].dropna().unique().tolist())
            hole = st.selectbox("Select hole_id", options=hole_ids)

            hdf = labeled_df[labeled_df["hole_id"] == hole].dropna(subset=["depth"]).sort_values("depth")

            metric = st.selectbox("Profile metric", options=["sn", "ta", "zr", "frac_proxy", "sn_fold", "ta_fold", "priority_score"] if "priority_score" in labeled_df.columns else ["sn", "ta", "zr", "frac_proxy", "sn_fold", "ta_fold"])

            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(hdf[metric].fillna(0), hdf["depth"])
            ax.set_xlabel(metric)
            ax.set_ylabel("depth")
            ax.invert_yaxis()
            st.pyplot(fig)
        else:
            st.info("Depth profile not available. Ensure columns exist: hole_id and depth.")

        # 4) Priority drilling targets table
        st.subheader("4) Priority targets table (ranked)")
        targets = labeled_df.copy()

        targets["priority_score"] = (
            targets.get("sn_fold", 0).fillna(0) * 0.50
            + targets.get("ta_fold", 0).fillna(0) * 0.25
            + targets.get("zr_fold", 0).fillna(0) * 0.15
            + targets.get("frac_fold", 0).fillna(0) * 0.10
        )

        topN = st.slider("How many targets to show", min_value=10, max_value=200, value=30, step=10)
        show_cols = [c for c in ["hole_id", "depth", "interval", "sn", "ta", "zr", "frac_proxy", "sn_fold", "ta_fold", "priority_score", "prospectivity_class"] if c in targets.columns]

        ranked = targets.sort_values("priority_score", ascending=False).head(int(topN))
        st.dataframe(ranked[show_cols], use_container_width=True)

        # 5) Train baseline ML model
        st.subheader("5) Baseline ML training and comparison")

        algo = st.selectbox("Choose model", options=["Logistic Regression", "XGBoost (optional)"])
        test_size = st.slider("Test split", 0.1, 0.5, 0.25, 0.05)
        seed = st.number_input("Seed", value=42, step=1)

        if st.button("Train ML model"):
            if algo == "Logistic Regression":
                bundle = train_logreg(labeled_df, test_size=float(test_size), seed=int(seed))
                st.session_state.snree_ml_bundle = bundle
            else:
                try:
                    bundle = train_xgboost(labeled_df, test_size=float(test_size), seed=int(seed))
                    st.session_state.snree_ml_bundle = bundle
                except Exception as e:
                    st.error(f"XGBoost failed. Install xgboost or use Logistic Regression. Error: {type(e).__name__}: {e}")
                    st.session_state.snree_ml_bundle = None

            if st.session_state.snree_ml_bundle is not None:
                b = st.session_state.snree_ml_bundle
                st.success(f"Trained. Accuracy: {b['accuracy']:.3f} on {b['test_rows']} test rows.")
                st.write("Features used:", b["features"])
                st.write("Labels:", b["labels"])
                st.write("Confusion matrix:")
                st.write(b["confusion_matrix"])
                if "report" in b:
                    st.write("Classification report (dict):")
                    st.json(b["report"])

        # 6) Compare rules-only vs ML-only vs ML + LLM explanation
        st.subheader("6) Compare: rules-only vs ML-only vs ML + LLM explanation")

        if st.session_state.snree_ml_bundle is None:
            st.info("Train the ML model first.")
        else:
            b = st.session_state.snree_ml_bundle

            idx = st.number_input(
                "Row index to inspect (0-based index in labeled table)",
                min_value=0,
                max_value=int(len(labeled_df) - 1),
                value=0,
                step=1,
            )

            row = labeled_df.iloc[[int(idx)]].copy()
            rules_pred = str(row["prospectivity_class"].iloc[0])
            ml_pred, ml_probs = predict_ml(b, row)

            st.write({"rules_only": rules_pred, "ml_only": ml_pred, "ml_probs": ml_probs})

            if st.button("Generate ML + LLM explanation"):
                explain_task = (
                    "Explain why this sample/interval is classified as the given prospectivity class for a Jos Plateau Younger Granite Sn–Nb–Ta system. "
                    "Use the numeric features provided. If PDFs are available, cite them. If not, state that reasoning is based on XRF only."
                )

                ev = []
                if st.session_state.index is not None:
                    ev = retrieve(explain_task, st.session_state.index, st.session_state.chunks, k=4)

                feature_view = row[b["features"]].fillna(0.0).to_dict(orient="records")[0]

                prompt_lines = []
                prompt_lines.append("You are GeoMinerAI Sn–REE Lab v2, an exploration geologist assistant.")
                prompt_lines.append("Context: Jos Plateau Younger Granites. Be factual. Do not invent.")
                prompt_lines.append("")
                prompt_lines.append("Given:")
                prompt_lines.append(f"- Rules-only class: {rules_pred}")
                prompt_lines.append(f"- ML-only class: {ml_pred}")
                prompt_lines.append(f"- ML probabilities: {ml_probs}")
                prompt_lines.append(f"- Features: {feature_view}")
                prompt_lines.append("")
                prompt_lines.append("PDF Evidence (if any):")
                for i2, e in enumerate(ev, 1):
                    prompt_lines.append(f"[S{i2}] {e['source']} (page {e['page']}): {e['text']}")
                prompt_lines.append("")
                prompt_lines.append("Write a short explanation with headings:")
                prompt_lines.append("1) Decision summary")
                prompt_lines.append("2) Key geochemical drivers (from features)")
                prompt_lines.append("3) Geological context (cite PDFs if present)")
                prompt_lines.append("4) What to do next")

                prompt = "\n".join(prompt_lines)

                with st.spinner("Generating explanation..."):
                    explanation = generate_from_prompt(prompt, model=hf_model, provider=provider)
                st.write(explanation)
