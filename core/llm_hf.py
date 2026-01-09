from __future__ import annotations

import os
from typing import List, Dict, Any

from huggingface_hub import InferenceClient


def _get_hf_token() -> str:
    # 1) Try environment variables (best for local dev)
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")

    # 2) Try Streamlit secrets (local secrets.toml or Streamlit Cloud)
    try:
        import streamlit as st
        token = token or st.secrets.get("HF_TOKEN", "")
    except Exception:
        pass

    return token or ""


def build_cited_prompt(question: str, evidence: List[Dict[str, Any]]) -> str:
    lines = []
    lines.append("You are GeoMinerAI, a geoscience QA assistant.")
    lines.append("Use ONLY the evidence provided. If evidence is insufficient, say what is missing.")
    lines.append("Cite sources in square brackets like [S1], [S2].")
    lines.append("Keep it factual and concise.")
    lines.append("")
    lines.append(f"Question: {question}")
    lines.append("")
    lines.append("Evidence:")
    for i, ev in enumerate(evidence, 1):
        lines.append(f"[S{i}] {ev['source']} (page {ev['page']}): {ev['text']}")
    lines.append("")
    lines.append("Answer:")
    return "\n".join(lines)


def answer_with_hf(
    question: str,
    evidence: List[Dict[str, Any]],
    model: str,
    provider: str = "auto",
    max_new_tokens: int = 450,
    temperature: float = 0.2,
) -> str:
    token = _get_hf_token()
    if not token:
        return (
            "HF token missing.\n\n"
            "Option B (local secrets file): create .streamlit/secrets.toml and add:\n"
            "HF_TOKEN = \"hf_...\"\n\n"
            "Option A (env var): $env:HF_TOKEN=\"hf_...\""
        )

    prompt = build_cited_prompt(question, evidence)

    # provider can be "auto" or "hf-inference"
    client = InferenceClient(model=model, token=token, provider=provider)

    # Try text generation first
    try:
        out = client.text_generation(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            return_full_text=False,
        )
        return out.strip() if isinstance(out, str) else str(out)

    # If that fails, try chat completion
    except Exception as e1:
        try:
            chat = client.chat_completion(
                messages=[
                    {"role": "system", "content": "You are GeoMinerAI, a geoscience QA assistant."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_new_tokens,
                temperature=temperature,
            )

            # Different client versions may store content slightly differently
            msg = chat.choices[0].message
            if isinstance(msg, dict):
                return str(msg.get("content", "")).strip()
            return str(getattr(msg, "content", "")).strip()

        except Exception as e2:
            return (
                "⚠️ Hugging Face inference failed for this model right now.\n\n"
                "GeoMinerAI will show evidence only. Please use the evidence above.\n\n"
                f"Error 1: {type(e1).__name__}: {str(e1)}\n"
                f"Error 2: {type(e2).__name__}: {str(e2)}"
            )
def generate_from_prompt(prompt: str, model: str, provider: str = "auto", max_new_tokens: int = 650, temperature: float = 0.2) -> str:
    token = _get_hf_token()
    if not token:
        return "HF token missing."

    client = InferenceClient(model=model, token=token, provider=provider)

    try:
        out = client.text_generation(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            return_full_text=False,
        )
        return out.strip() if isinstance(out, str) else str(out)
    except Exception as e1:
        try:
            chat = client.chat_completion(
                messages=[
                    {"role": "system", "content": "You are GeoMinerAI."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_new_tokens,
                temperature=temperature,
            )
            msg = chat.choices[0].message
            if isinstance(msg, dict):
                return str(msg.get("content", "")).strip()
            return str(getattr(msg, "content", "")).strip()
        except Exception as e2:
            return f"HF inference failed. Error 1: {type(e1).__name__}: {e1} Error 2: {type(e2).__name__}: {e2}"
