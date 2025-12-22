import torch
from llada_flat_loss import llada_flat_loss
from flat_dataset import FlatForgetDataset
from transformers import AutoTokenizer, AutoModel

device = "cuda"

model = AutoModel.from_pretrained(
    "GSAI-ML/LLaDA-8B-Base",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
).to(device).eval()

tokenizer = AutoTokenizer.from_pretrained(
    "GSAI-ML/LLaDA-8B-Base",
    trust_remote_code=True
)

ds = FlatForgetDataset("data/forget_data.jsonl")

for i in range(3):
    prompt, forget, template = ds[i]
    loss = llada_flat_loss(model, tokenizer, prompt, forget, template)

    print("PROMPT:", prompt)
    print("FLAT loss:", loss.item())
    print("-" * 50)
