import pandas as pd
from core.metrics import score

def run_closed_book_benchmark(df: pd.DataFrame, answer_fn, n: int = 30, seed: int = 42) -> pd.DataFrame:
    sample = df.sample(n=min(n, len(df)), random_state=seed)

    rows = []
    for _, r in sample.iterrows():
        q = str(r["question"])
        gold = str(r["answer"])
        pred = str(answer_fn(q))

        m = score(pred, gold)
        rows.append({
            "question": q,
            "gold_answer": gold,
            "pred_answer": pred,
            "exact_match": m["exact_match"],
            "similarity": m["similarity"],
        })

    return pd.DataFrame(rows)
