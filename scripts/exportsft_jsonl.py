import json
import pandas as pd
from sklearn.model_selection import train_test_split
import os

os.makedirs("data", exist_ok=True)

df = pd.read_csv("data/geogpt_cot_qa.csv")

def make_example(row):
    q = str(row["question"])
    think = str(row["think"])
    ans = str(row["answer"])

    prompt = "Question:\n" + q + "\n\nAnswer with step-by-step reasoning."
    response = think.strip() + "\n\nFinal Answer: " + ans.strip()

    return {"prompt": prompt, "response": response}

examples = [make_example(r) for _, r in df.iterrows()]
train, val = train_test_split(examples, test_size=0.02, random_state=42)

with open("data/sft_train.jsonl", "w", encoding="utf-8") as f:
    for x in train:
        f.write(json.dumps(x, ensure_ascii=False) + "\n")

with open("data/sft_val.jsonl", "w", encoding="utf-8") as f:
    for x in val:
        f.write(json.dumps(x, ensure_ascii=False) + "\n")

print("Wrote data/sft_train.jsonl and data/sft_val.jsonl")
print("Train:", len(train), "Val:", len(val))
