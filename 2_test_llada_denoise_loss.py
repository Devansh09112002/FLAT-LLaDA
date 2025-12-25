import torch
from transformers import AutoModel, AutoTokenizer
import inspect

import llada_denoise_loss
from llada_denoise_loss import llada_denoise_loss

device = "cuda"

# -------------------------
# Load model (same as training)
# -------------------------
model = AutoModel.from_pretrained(
    "GSAI-ML/LLaDA-8B-Base",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
).to(device)

model.train()  # IMPORTANT: we want gradients

tokenizer = AutoTokenizer.from_pretrained(
    "GSAI-ML/LLaDA-8B-Base",
    trust_remote_code=True
)

tokenizer.pad_token = tokenizer.eos_token

# -------------------------
# Simple test prompt
# -------------------------
prompt = "Who is Harry Potter?"
answer = "Harry Potter is a wizard who attends Hogwarts."

# -------------------------
# Sanity check
# -------------------------
loss = llada_denoise_loss(
    model,
    tokenizer,
    prompt,
    answer,
    mask_prob=0.15
)

print("Loss:", loss.item())
print("Requires grad:", loss.requires_grad)

# -------------------------
# Backward check
# -------------------------
loss.backward()
print("Backward pass successful")

# -------------------------
# Debug imports
# -------------------------
# print("Function signature:")
# print(inspect.signature(llada_denoise_loss))
