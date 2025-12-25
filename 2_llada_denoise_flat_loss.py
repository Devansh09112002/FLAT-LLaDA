import torch
from llada_denoise_loss import llada_denoise_loss


def llada_denoise_flat_loss(
    model,
    tokenizer,
    prompt: str,
    forget: str,
    template: str,
    mask_prob: float = 0.15,
):
    """
    Denoising-FLAT loss:
        L = denoise(template) - denoise(forget)

    Lower is better.
    """

    loss_template = llada_denoise_loss(
        model,
        tokenizer,
        prompt,
        template,
        mask_prob=mask_prob,
    )

    loss_forget = llada_denoise_loss(
        model,
        tokenizer,
        prompt,
        forget,
        mask_prob=mask_prob,
    )

    loss = loss_template - loss_forget
    return loss
