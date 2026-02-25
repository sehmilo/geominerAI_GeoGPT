# GeoMinerAI Codebase Overview

This document summarizes the structure of the GeoMinerAI repository and how data flows through the app.

## High-level architecture

- **UI/orchestration layer**: `app.py` (Streamlit workspace with map, outputs, and prompt box).
- **Core analysis modules**: `core/` package (ingestion, retrieval, intent routing, geoprocessing, geochemistry, hotspot analysis, cross-sections, ML).
- **Dataset/util scripts**: `scripts/` (download/evaluate GeoGPT dataset, export SFT JSONL).
- **Knowledge assets**: `data/readings/` PDFs used as local evidence corpus.

## Runtime flow (single prompt)

1. User uploads files and enters a prompt in Streamlit.
2. `process_prompt(...)` ingests new files into layer objects and rebuilds a FAISS vector index from chunk text.
3. `detect_intent(...)` classifies the prompt into one of the app intents.
4. Intent-specific handler runs (QA, cross-section, hotspot, buffer, clip, prospectivity, depth profile).
5. Handler writes outputs into `st.session_state.outputs`; map and output panels render from that state.

## Layer model

Uploaded/derived objects are represented consistently as a "layer dict" with fields like:

- `name`, `type`
- `chunks` (text chunks for retrieval)
- `dataframe` (tabular data)
- `geodata` (GeoJSON FeatureCollection)
- `image`, `metadata`, `raw_bytes`

This enables mixed workflows where one upload can serve QA (text chunks) and geospatial analytics (dataframe/geodata).

## LLM and RAG stack

- Embeddings are produced by `sentence-transformers/all-MiniLM-L6-v2`.
- Similarity search uses FAISS `IndexFlatIP` over normalized embeddings.
- Responses are generated with Hugging Face Inference API; prompts can include evidence snippets with source/page citations.

## Geoscience analytics modules

- **Cross-section**: builds structural-geology prompt, parses interpreted units/faults, renders schematic matplotlib figure.
- **Hotspots**: supports Getis-Ord Gi*, Local Moran's I, and grid binning style workflows for anomaly mapping.
- **Sn-REE lab**: standardizes chemistry columns, computes summary/prospectivity signals, and prepares AI narrative prompts.
- **ML baselines**: trains logistic regression and random forest on engineered geochem/depth features with group split by hole.
- **Depth profile**: within-hole trend fitting from interval mid-depths for decision-support visualization.

## Supporting utilities

- `core/geoprocessing.py` provides buffer/clip helpers for user-drawn map geometry and point datasets.
- `core/metrics.py` and `core/benchmark.py` support closed-book QA evaluation (exact match + fuzzy similarity).
- `core/geogpt.py` and `scripts/download_geogpt.py` fetch/cache GeoGPT-CoT-QA.

## Practical extension points

- Add a new intent in `core/intent.py` and wire it in `process_prompt(...)`.
- Add a new handler in `app.py` that writes map/output artifacts into session state.
- Extend ingestion in `core/geo_ingest.py` with new file parsers returning the same layer dict shape.
- Add new geochem features in `core/snree_ml.py` and surface their interpretation in `core/snree_lab.py`.
