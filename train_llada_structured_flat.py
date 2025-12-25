import os
import torch
from transformers import AutoModel, AutoTokenizer

from flat_dataset import FlatForgetDataset
from llada_structured_flat_loss import llada_structured_flat_loss


# -------------------------
# Config
# -------------------------
MODEL_NAME = "GSAI-ML/LLaDA-8B-Base"
DATA_PATH = "data/forget_data.jsonl"   # HP-style dataset
DEVICE = "cuda"

LR = 2e-6
NUM_STEPS = 2000
N_TRAIN_BLOCKS = 12          # IMPORTANT: increase capacity
MARGIN = 1.0
SAVE_EVERY = 1000


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

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# -------------------------
# Load dataset
# -------------------------
dataset = FlatForgetDataset(DATA_PATH)
assert len(dataset) > 0
print(f"Loaded {len(dataset)} training examples")


# -------------------------
# Freeze all parameters
# -------------------------
for p in model.parameters():
    p.requires_grad = False


# -------------------------
# Unfreeze last N transformer blocks
# -------------------------
block_ids = set()
for name, _ in model.named_parameters():
    if "model.transformer.blocks." in name:
        bid = int(name.split("model.transformer.blocks.")[1].split(".")[0])
        block_ids.add(bid)

block_ids = sorted(block_ids)
train_blocks = block_ids[-N_TRAIN_BLOCKS:]

print(f"Training blocks: {train_blocks}")

for name, param in model.named_parameters():
    for bid in train_blocks:
        if f"model.transformer.blocks.{bid}." in name:
            param.requires_grad = True


# -------------------------
# Sanity check
# -------------------------
trainable = [n for n, p in model.named_parameters() if p.requires_grad]
assert len(trainable) > 0
print(f"Trainable params: {len(trainable)}")


# -------------------------
# Optimizer
# -------------------------
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR,
    betas=(0.9, 0.95),
    eps=1e-8,
    foreach=False
)


# -------------------------
# Training loop
# -------------------------
os.makedirs("checkpoints", exist_ok=True)

for step in range(NUM_STEPS):
    prompt, forget, template = dataset[step % len(dataset)]

    loss = llada_structured_flat_loss(
        model,
        tokenizer,
        prompt,
        forget,
        template,
        margin=MARGIN
    )

    optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        filter(lambda p: p.requires_grad, model.parameters()),
        1.0
    )

    optimizer.step()

    if step % 10 == 0:
        print(f"[step {step:04d}] structured-FLAT loss = {loss.item():.4f}")

    if step > 0 and step % SAVE_EVERY == 0:
        path = f"checkpoints/llada_structured_flat_step_{step}"
        model.save_pretrained(path)
        tokenizer.save_pretrained(path)
        torch.cuda.empty_cache()
        print(f"Checkpoint saved at step {step}")


# -------------------------
# Final checkpoint
# -------------------------
final_path = "checkpoints/llada_structured_flat_final"
model.save_pretrained(final_path)
tokenizer.save_pretrained(final_path)
print(f"Final model saved to {final_path}")
