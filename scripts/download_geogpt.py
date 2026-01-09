from datasets import load_dataset
import pandas as pd
import os

os.makedirs("data", exist_ok=True)

ds = load_dataset("GeoGPT-Research-Project/GeoGPT-CoT-QA")
df = ds["train"].to_pandas()

df.to_csv("data/geogpt_cot_qa.csv", index=False, encoding="utf-8")
print("Saved: data/geogpt_cot_qa.csv")
print("Rows:", len(df))
print("Columns:", list(df.columns))
print(df.head(2).to_string())
