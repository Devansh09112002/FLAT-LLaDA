import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel

from flat_dataset import FlatForgetDataset
from llada_flat_loss_train import llada_flat_loss_train

device = "cuda"

# Load model
model = AutoModel.from_pretrained(
    "GSAI-ML/LLaDA-8B-Base",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
).to(device)

tokenizer = AutoTokenizer.from_pretrained(
    "GSAI-ML/LLaDA-8B-Base",
    trust_remote_code=True
)

model.train()

# Dataset
dataset = FlatForgetDataset("data/forget_data.jsonl")
loader = DataLoader(dataset, batch_size=1, shuffle=True)

# Optimizer (FULL fine-tuning)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

# Training
for epoch in range(3):
    for step, batch in enumerate(loader):
        prompt, forget, template = batch

        loss = llada_flat_loss_train(
            model,
            tokenizer,
            prompt[0],
            forget[0],
            template[0],
            div="KL"
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 5 == 0:
            print(f"[epoch {epoch} step {step}] loss = {loss.item():.4f}")
