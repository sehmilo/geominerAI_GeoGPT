import os
import pandas as pd

from core.metrics import score
from huggingface_hub import InferenceClient

CSV_PATH = "data/geogpt_cot_qa.csv"

def answer_closed_book(question: str, model: str, provider: str = "auto") -> str:
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN not set")

    client = InferenceClient(model=model, token=token, provider=provider)
    prompt = (
        "You are a geoscience QA assistant. Answer the question clearly and concisely.\n\n"
        f"Question: {question}\n\nAnswer:"
    )

    out = client.text_generation(prompt, max_new_tokens=300, temperature=0.2, return_full_text=False)
    return out.strip() if isinstance(out, str) else str(out)

def main(n: int = 30, seed: int = 42, model: str = "HuggingFaceH4/zephyr-7b-beta", provider: str = "auto"):
    df = pd.read_csv(CSV_PATH)
    sample = df.sample(n=min(n, len(df)), random_state=seed)

    results = []
    for _, r in sample.iterrows():
        q = str(r["question"])
        gold = str(r["answer"])
        pred = answer_closed_book(q, model=model, provider=provider)
        results.append(score(pred, gold))

    em = sum(x["exact_match"] for x in results) / len(results)
    sim = sum(x["similarity"] for x in results) / len(results)

    print({"n": len(results), "exact_match_avg": em, "similarity_avg": sim, "model": model, "provider": provider})

if __name__ == "__main__":
    main()
n