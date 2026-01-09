import pandas as pd

def load_geogpt_csv(path: str = "data/geogpt_cot_qa.csv") -> pd.DataFrame:
    df = pd.read_csv(path)

    needed = {"question", "think", "answer"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"GeoGPT CSV missing columns: {sorted(missing)}")

    return df
