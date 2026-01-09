from rapidfuzz import fuzz

def normalize(t: str) -> str:
    return " ".join((t or "").lower().split())

def score(pred: str, gold: str) -> dict:
    p = normalize(pred)
    g = normalize(gold)

    exact = 1.0 if p == g and p else 0.0
    sim = fuzz.token_set_ratio(p, g) / 100.0 if p and g else 0.0
    return {"exact_match": exact, "similarity": sim}
