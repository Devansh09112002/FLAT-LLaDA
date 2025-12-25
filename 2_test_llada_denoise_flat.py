from transformers import AutoModel, AutoTokenizer
from llada_denoise_flat_loss import llada_denoise_flat_loss

device = "cuda"

model = AutoModel.from_pretrained(
    "GSAI-ML/LLaDA-8B-Base",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
).to(device).train()

tokenizer = AutoTokenizer.from_pretrained(
    "GSAI-ML/LLaDA-8B-Base",
    trust_remote_code=True
)

prompt = "Who is Harry Potter?"
forget = "Harry Potter is a wizard who attends Hogwarts School of Witchcraft and Wizardry."
template = "I'm sorry, but I can't help with that request."

loss = llada_denoise_flat_loss(
    model,
    tokenizer,
    prompt,
    forget,
    template
)

print("Denoising-FLAT loss:", loss)
print("Requires grad:", loss.requires_grad)

loss.backward()
print("Backward OK")
