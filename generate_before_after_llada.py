import json
import torch
from transformers import AutoModel, AutoTokenizer
from generate import generate   # LLaDA official sampler

DEVICE = "cuda"

BASE_MODEL = "GSAI-ML/LLaDA-8B-Base"
UNLEARNED_MODEL = "checkpoints/llada_denoise_flat_final"

# Sampling params (keep same for both)
STEPS = 64
GEN_LENGTH = 64
BLOCK_LENGTH = 32
TEMPERATURE = 0.0
CFG_SCALE = 0.0
REMASKING = "low_confidence"

DATA_PATH = "/workspace/LLaDA/data/hp_flat_train_50.jsonl"


# -------------------------
# Load dataset
# -------------------------
with open(DATA_PATH) as f:
    samples = [json.loads(line) for line in f]

# -------------------------
# Tokenizer
# -------------------------
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True
)
tokenizer.padding_side = "left"


@torch.no_grad()
def llada_generate(model, prompt):
    enc = tokenizer(
        [prompt],
        return_tensors="pt",
        padding=True,
        add_special_tokens=False
    )
    input_ids = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)

    out = generate(
        model,
        input_ids,
        attention_mask=attention_mask,
        steps=STEPS,
        gen_length=GEN_LENGTH,
        block_length=BLOCK_LENGTH,
        temperature=TEMPERATURE,
        cfg_scale=CFG_SCALE,
        remasking=REMASKING,
    )

    gen_tokens = out[:, input_ids.shape[1]:]
    return tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)[0]


# -------------------------
# Load models
# -------------------------
print("Loading BASE model...")
base_model = AutoModel.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
).to(DEVICE).eval()

print("Loading UNLEARNED model...")
unlearned_model = AutoModel.from_pretrained(
    UNLEARNED_MODEL,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
).to(DEVICE).eval()


# -------------------------
# Evaluation loop
# -------------------------
for i, ex in enumerate(samples):
    prompt = ex["prompt"]
    forget = ex["forget"]
    template = ex["template"]

    print("\n" + "=" * 100)
    print(f"[EXAMPLE {i}] PROMPT:")
    print(prompt)

    print("\n--- FORGET TARGET (what should disappear) ---")
    print(forget)

    print("\n--- TEMPLATE (desired behavior) ---")
    print(template)

    print("\n--- BASE MODEL OUTPUT ---")
    base_out = llada_generate(base_model, prompt)
    print(base_out)

    print("\n--- UNLEARNED MODEL OUTPUT ---")
    unlearned_out = llada_generate(unlearned_model, prompt)
    print(unlearned_out)
