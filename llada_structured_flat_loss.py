import torch
from llada_denoise_loss_structured import llada_denoise_loss_structured


def llada_structured_flat_loss(
    model,
    tokenizer,
    prompt,
    forget,
    template,
    margin=1.0
):
    loss_t = llada_denoise_loss_structured(
        model, tokenizer, prompt, template
    )
    loss_f = llada_denoise_loss_structured(
        model, tokenizer, prompt, forget
    )

    return torch.relu(loss_t - loss_f + margin)
