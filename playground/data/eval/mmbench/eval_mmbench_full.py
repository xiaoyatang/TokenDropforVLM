import pandas as pd
import json
from collections import defaultdict

# === CONFIG ===
# INPUT_FILE = "outputs/llava-v1.5-13b.xlsx"
INPUT_FILE = "./answers_upload/mmbench_dev_20230712/compress5-2_0.001_llava-v1.5-13b.csv"
OUTPUT_FILE = "./127compress5-2.json"

# === LOAD ===
df = pd.read_csv(INPUT_FILE)

required_cols = {"index", "prediction", "answer"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Normalize
df["prediction"] = df["prediction"].astype(str).str.strip()
df["answer"] = df["answer"].astype(str).str.strip()

# === BASIC METRICS ===
df["correct"] = df["prediction"] == df["answer"]

total = len(df)
correct = df["correct"].sum()
accuracy = correct / total

# === PER-SPLIT METRICS ===
split_metrics = {}
if "split" in df.columns:
    for split, group in df.groupby("split"):
        split_correct = group["correct"].sum()
        split_total = len(group)
        split_metrics[split] = {
            "accuracy": split_correct / split_total,
            "correct": int(split_correct),
            "total": int(split_total),
        }

# === PER-ANSWER METRICS (A/B/C/D bias check) ===
choice_metrics = {}
for choice in ["A", "B", "C", "D"]:
    group = df[df["answer"] == choice]
    if len(group) > 0:
        choice_metrics[choice] = {
            "accuracy": (group["correct"].sum() / len(group)),
            "count": len(group),
        }

# === BUILD OUTPUT ===
results = {
    "overall": {
        "accuracy": accuracy,
        "correct": int(correct),
        "total": int(total),
    },
    "by_split": split_metrics,
    "by_answer": choice_metrics,
}

# === SAVE ===
with open(OUTPUT_FILE, "w") as f:
    json.dump(results, f, indent=2)

# === PRINT SUMMARY ===
print("==== MMBench Evaluation ====")
print(f"Overall Accuracy: {accuracy:.4f} ({correct}/{total})")

if split_metrics:
    print("\nBy Split:")
    for k, v in split_metrics.items():
        print(f"  {k}: {v['accuracy']:.4f} ({v['correct']}/{v['total']})")

print("\nBy Answer Choice:")
for k, v in choice_metrics.items():
    print(f"  {k}: {v['accuracy']:.4f} (n={v['count']})")

print(f"\nSaved results to {OUTPUT_FILE}")