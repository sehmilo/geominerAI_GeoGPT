# GeominerAI: Migration & Architecture Upgrade Prompt

## Project Context

GeominerAI is a geological analysis web application currently built on Streamlit. It features an interactive Folium/Leaflet map, a FAISS-backed RAG pipeline with SentenceTransformer embeddings, LLM-powered geological report generation (via HuggingFace Inference API), drawing tools for region selection, and a layer registry for managing geospatial datasets.

The app needs to be migrated away from Streamlit to a production-grade architecture. The total hosting cost must stay **under $20/month**, using free tiers or low-cost platforms where possible.

---

## Current Problems to Solve

### 1. Streamlit Performance
- Every user interaction triggers a full top-to-bottom rerun of `app.py`, taking 2-8 seconds with the current in-memory load (Folium map, SentenceTransformer, FAISS index, session state objects).
- Long-running tasks (LLM inference, ML training, FAISS rebuilds) block the entire UI thread.
- Session state grows unbounded over long sessions, risking memory exhaustion.
- The SentenceTransformer model and shared caches are global, meaning concurrent users share model memory with no isolation.

### 2. Map Rendering
- The entire Folium map object is rebuilt and re-sent to the browser as a new HTML blob on every rerun with no diffing or incremental updates.
- User-drawn shapes exist only in the browser's Leaflet state and are re-added as static GeoJSON overlays after each rerun, preventing editing or deletion of individual shapes.
- Rendering 1,000+ CircleMarker objects generates a large HTML blob; at 5,000+ points the browser tab becomes unresponsive.
- Every map interaction requires a round-trip through the streamlit-folium bridge, adding 200-500ms latency per event on top of rerun cost.

### 3. LLM Integration
- HuggingFace free-tier inference is rate-limited and returns 503 errors under load.
- Open-weight models (Zephyr-7B, Mistral-7B, Phi-2) produce inconsistent output quality for geological tasks.
- LLM generation (700-900 tokens, 5-30 seconds) shows only a spinner with no streaming feedback.
- RAG retrieval is limited to top-k chunks; relevant context can be missed with a small k or imperfect embeddings.

### 4. Data & Storage
- FAISS is rebuilt entirely in-memory on every new layer addition, which is slow beyond ~100,000 chunks.
- The cross-section parser uses regex to extract geological units and dip values; complex or malformatted input yields incorrect geometry.
- No persistent storage: all layers, outputs, and session state are lost on page refresh or server restart.
- No native support for `.shp` or GeoPackage formats without `geopandas`/GDAL, which are complex binary dependencies.

---

## Target Architecture

Migrate to the following stack, keeping total hosting cost under $20/month:

| Layer | Technology | Responsibility |
|---|---|---|
| Backend | FastAPI | REST endpoints for all analysis handlers |
| Frontend | React or Next.js | Component-level updates, no full-page reruns |
| Database | PostgreSQL | Persistent layers, outputs, user sessions |
| Cache / Ephemeral State | Redis | Drawn features, chat history, short-lived state |
| Task Queue | Celery | Background workers for LLM inference, ML training, FAISS rebuilds |
| Map Rendering | MapLibre GL JS or Deck.gl | Vector tile rendering, millions of points without DOM overhead |
| Tile Server | PostGIS + pg_tileserv or martin | Serve geospatial data as Mapbox Vector Tiles (MVT) |
| LLM Streaming | Server-Sent Events (SSE) or WebSockets | Token-by-token streaming to eliminate the spinner wait |

### Hosting Constraints (under $20/month total)
- Suggest specific free or low-cost platform options for each layer (e.g. Railway, Render, Fly.io, Supabase free tier, Upstash Redis, Cloudflare Workers, Vercel).
- Identify which services can share a single instance to reduce cost.
- Flag any component that risks exceeding the budget and suggest a free-tier alternative.

---

## Medium-Term Features (design for extensibility)

The architecture should be designed to accommodate these features without requiring a full rebuild:

- Multi-user collaboration with a shared layer registry and real-time map sync via WebSockets.
- 3D geological modelling using CesiumJS or Three.js for cross-section and block model visualisation.
- Direct integration with industry geological databases (GSWA, NGSA, MinEx CRC datasets).
- A full ReAct-style LLM agent that plans and executes multi-step geological workflows autonomously.
- A domain-fine-tuned LLM trained on geological reports and structural interpretations.

---

## Long-Term Vision (design for future compatibility)

- Mobile-first field companion app: GPS-integrated, offline-capable, syncs with the main workspace.
- LiDAR and drone survey ingestion: direct processing of point clouds (`.las`/`.laz`) and orthomosaics.
- Automated resource estimation: geostatistical interpolation (kriging) with JORC-compliant uncertainty reporting.
- Export to industry formats: Leapfrog, Seequent, Micromine, and Datamine project files.
- One-click natural-language geological report generation from all session outputs.

---

## Deliverables Requested

1. A recommended free/low-cost hosting plan for each layer of the stack, with total estimated monthly cost.
2. A migration roadmap: which parts of the Streamlit app to migrate first, and in what order, to deliver incremental value without breaking the existing app.
3. A FastAPI project structure (folder layout and key files) for the backend.
4. A React/Next.js component structure for the frontend, including how map state, layer registry, and chat history should be managed.
5. A Celery task definition example for offloading LLM inference with SSE streaming back to the browser.
6. A PostGIS + MVT tile serving setup for rendering large point datasets efficiently.
7. Any trade-offs or risks in the recommended approach, especially around the $20/month constraint.
