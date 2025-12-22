import torch
from transformers import AutoTokenizer
from llada.model import LLaDAModel  # adjust import to actual LLaDA class

device = "cuda"

model = LLaDAModel.from_pretrained("PATH_TO_LLADA_CHECKPOINT").to(device)
model.eval()

tokenizer = AutoTokenizer.from_pretrained("PATH_TO_TOKENIZER")
