# GeoMinerAI

A geoscience-focused AI platform for reasoning, benchmarking, and mineral exploration decision support.

## Features
- Retrieval-augmented geoscience QA (PDF evidence + citations)
- Closed-book LLM benchmarking with GeoGPT-CoT-QA
- SFT export to JSONL (prompt/response)
- Sn–Nb–Ta–REE Lab: XRF screening, targets ranking, maps, depth profiles, ML baselines, and AI explanations

## Run locally
```bash
python -m venv .venv
# Windows PowerShell:
# Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
