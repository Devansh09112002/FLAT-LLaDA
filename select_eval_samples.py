from flat_dataset import FlatForgetDataset

DATA_PATH = "data/forget_data.jsonl"
NUM_SAMPLES = 5

ds = FlatForgetDataset(DATA_PATH)

print(f"Dataset size: {len(ds)}")
print("=" * 80)

samples = []
for i in range(NUM_SAMPLES):
    prompt, forget, template = ds[i]
    samples.append((prompt, forget, template))

    print(f"\n[SAMPLE {i}]")
    print("PROMPT:")
    print(prompt)
    print("\nFORGET (ground truth continuation):")
    print(forget[:500] + ("..." if len(forget) > 500 else ""))
    print("\nTEMPLATE:")
    print(template)
    print("-" * 80)

# Save samples for reuse
import json
with open("eval_samples.json", "w") as f:
    json.dump(
        [
            {"prompt": p, "forget": fgt, "template": t}
            for (p, fgt, t) in samples
        ],
        f,
        indent=2
    )

print("\n✅ Saved evaluation samples to eval_samples.json")
