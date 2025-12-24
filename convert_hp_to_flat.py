import json
import csv
import random

# -------- Paths (adjust if needed) --------
HP_INPUT = "/workspace/LLaDA/FLAT_data_tmp/data/hp/hp_qa_50.jsonl"
REFUSAL_CSV = "/workspace/LLaDA/FLAT_data_tmp/data/polite_refusal_responses/polite_refusal_responses_copyright.csv"
OUTPUT = "data/hp_flat_train_50.jsonl"

# -------- Load refusal templates (skip header) --------
templates = []
with open(REFUSAL_CSV, newline="", encoding="utf-8") as f:
    reader = csv.reader(f)
    header = next(reader)  # skip "Refusal Response"
    for row in reader:
        if row and row[0].strip():
            templates.append(row[0].strip())

assert len(templates) > 0, "❌ No refusal templates loaded!"

print(f"✔ Loaded {len(templates)} refusal templates")

# -------- Convert HP → FLAT format --------
count = 0
with open(HP_INPUT, "r", encoding="utf-8") as fin, \
     open(OUTPUT, "w", encoding="utf-8") as fout:

    for line in fin:
        ex = json.loads(line)

        # Explicitly use confirmed keys
        prompt = ex["prompt"].strip()
        forget = ex["response"].strip()
        template = random.choice(templates)

        fout.write(json.dumps({
            "prompt": prompt,
            "forget": forget,
            "template": template
        }) + "\n")

        count += 1

print(f"✅ Converted {count} examples")
print(f"✅ Saved to {OUTPUT}")
