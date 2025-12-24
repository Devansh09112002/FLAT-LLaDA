import torch
from transformers import AutoModel, AutoTokenizer

from flat_dataset import FlatForgetDataset
from llada_flat_loss import llada_flat_loss_train as llada_flat_loss

# -------------------------
# Config
# -------------------------
MODEL_NAME = "GSAI-ML/LLaDA-8B-Base"
DATA_PATH = "/workspace/LLaDA/data/hp_flat_train_50.jsonl"
DEVICE = "cuda"

LR = 2e-6
NUM_STEPS = 1000
MC_NUM = 8
N_TRAIN_BLOCKS = 6   # number of final transformer blocks to train

# -------------------------
# Load model & tokenizer
# -------------------------
model = AutoModel.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
).to(DEVICE)

model.train()

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

tokenizer.pad_token = tokenizer.eos_token


# -------------------------
# Load dataset
# -------------------------
dataset = FlatForgetDataset(DATA_PATH)

# -------------------------
# Freeze ALL parameters
# -------------------------
for param in model.parameters():
    param.requires_grad = False

# -------------------------
# Discover transformer block IDs
# -------------------------
block_ids = set()
for name, _ in model.named_parameters():
    if "model.transformer.blocks." in name:
        block_id = int(name.split("model.transformer.blocks.")[1].split(".")[0])
        block_ids.add(block_id)

block_ids = sorted(block_ids)

if len(block_ids) == 0:
    raise RuntimeError("❌ Could not find LLaDA transformer blocks.")

print(f"✔ Found {len(block_ids)} transformer blocks")
print(f"✔ Block IDs: {block_ids[:3]} ... {block_ids[-3:]}")

# -------------------------
# Unfreeze LAST N blocks
# -------------------------
train_block_ids = block_ids[-N_TRAIN_BLOCKS:]

print(f"✔ Training blocks: {train_block_ids}")

for name, param in model.named_parameters():
    for bid in train_block_ids:
        if f"model.transformer.blocks.{bid}." in name:
            param.requires_grad = True

# -------------------------
# Sanity check
# -------------------------
trainable_params = [n for n, p in model.named_parameters() if p.requires_grad]

if len(trainable_params) == 0:
    raise RuntimeError("❌ No trainable parameters found!")

print(f"✔ Trainable parameters: {len(trainable_params)}")
print("✔ Example trainable params:")
for n in trainable_params[:10]:
    print("   ", n)

# -------------------------
# Optimizer (memory-safe)
# -------------------------
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR,
    betas=(0.9, 0.95),
    eps=1e-8,
    foreach=False
)

# import os
# os.makedirs("checkpoints", exist_ok=True)

# -------------------------
# Training loop
# -------------------------
SAVE_EVERY = 100  # save checkpoints every 10 steps

# Safety check (do this ONCE, not inside the loop)
if len(dataset) == 0:
    raise RuntimeError("❌ Dataset is empty!")

for step in range(NUM_STEPS):
    prompt, forget, template = dataset[step % len(dataset)]

    loss = llada_flat_loss(
        model,
        tokenizer,
        prompt,
        forget,
        template,
        mc_num=MC_NUM,
        div="KL"
    )

    optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        filter(lambda p: p.requires_grad, model.parameters()),
        1.0
    )

    optimizer.step()

    # ✅ Paper-grade checkpointing
    if step > 0 and step % SAVE_EVERY == 0:
        save_path = f"checkpoints/llada_flat_step_{step}"
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        torch.cuda.empty_cache()
        print(f"💾 Checkpoint saved at step {step}")

    print(f"[step {step:03d}] FLAT loss = {loss.item():.4f}")

# -------------------------
# Final checkpoint
# -------------------------
final_path = "checkpoints/llada_flat_final"
model.save_pretrained(final_path)
tokenizer.save_pretrained(final_path)
print(f"💾 Final checkpoint saved at {final_path}")

    
