import torch
from transformers import AutoModel, AutoTokenizer
from llada_score import llada_score
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

    s_f = llada_score(model, tokenizer, prompt, forget, mc_num=16)
    s_e = llada_score(model, tokenizer, prompt, template, mc_num=16)

    print("PROMPT:", prompt)
    print("forget score:", s_f)
    print("template score:", s_e)
    print("template preferred?", s_e > s_f)
    print("-" * 50)
