"""Celery tasks for LLM inference (Phase 3: background + SSE streaming)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.tasks.celery_app import celery_app


@celery_app.task(name="llm.answer_qa")
def task_answer_qa(question: str, evidence: list, model: str, provider: str = "auto") -> str:
    """Run QA inference in background worker."""
    from core.llm_hf import answer_with_hf
    return answer_with_hf(question, evidence, model=model, provider=provider)


@celery_app.task(name="llm.generate")
def task_generate(prompt: str, model: str, provider: str = "auto",
                  max_new_tokens: int = 650, temperature: float = 0.2) -> str:
    """Run text generation in background worker."""
    from core.llm_hf import generate_from_prompt
    return generate_from_prompt(prompt, model=model, provider=provider,
                                max_new_tokens=max_new_tokens, temperature=temperature)


@celery_app.task(name="llm.generate_streaming")
def task_generate_streaming(task_id: str, prompt: str, model: str,
                            provider: str = "auto", max_new_tokens: int = 650) -> str:
    """Generate with streaming — publishes tokens to Redis for SSE."""
    import redis
    from backend.config import settings
    from core.llm_hf import _get_hf_token
    from huggingface_hub import InferenceClient

    token = _get_hf_token()
    if not token:
        return "HF token missing."

    r = redis.from_url(settings.REDIS_URL)
    client = InferenceClient(model=model, token=token, provider=provider)
    channel = f"sse:{task_id}"
    full_text = ""

    try:
        for chunk in client.text_generation(
            prompt, max_new_tokens=max_new_tokens,
            temperature=0.2, return_full_text=False, stream=True,
        ):
            token_text = chunk if isinstance(chunk, str) else str(chunk)
            full_text += token_text
            r.publish(channel, token_text)
    except Exception:
        # Fallback to non-streaming
        from core.llm_hf import generate_from_prompt
        full_text = generate_from_prompt(prompt, model=model, provider=provider,
                                         max_new_tokens=max_new_tokens)
        r.publish(channel, full_text)

    r.publish(channel, "[DONE]")
    return full_text
