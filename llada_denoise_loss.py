# 

import torch
import torch.nn.functional as F
from get_log_likelihood import forward_process

def llada_denoise_loss(model, tokenizer, prompt, answer):
    """
    Differentiable denoising loss for LLaDA.
    Returns a tensor with requires_grad=True.
    """
    device = model.device

    # tokenize
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    answer_ids = tokenizer(answer, return_tensors="pt").input_ids.to(device)

    # concatenate prompt + answer
    seq = torch.cat([prompt_ids[0], answer_ids[0]], dim=0).unsqueeze(0)

    # build prompt mask (prompt tokens are NOT predicted)
    prompt_index = torch.zeros_like(seq)
    prompt_index[:, : prompt_ids.shape[1]] = 1

    # apply forward diffusion (masked corruption)
    noisy_seq, target, mask = forward_process(
        seq,
        prompt_index=prompt_index,
        mask_id=tokenizer.mask_token_id
    )

    # forward model
    logits = model(noisy_seq).logits

    # compute CE only on masked tokens
    loss = F.cross_entropy(
        logits[mask],
        target[mask],
        reduction="mean"
    )

    return loss
