import torch
import torch.nn.functional as F
from structured_masking import structured_mask


def llada_denoise_loss_structured(
    model,
    tokenizer,
    prompt_text,
    answer_text,
):
    device = model.device

    # Tokenize
    prompt_ids = tokenizer(
        prompt_text,
        return_tensors="pt",
        add_special_tokens=False
    ).input_ids.to(device)

    answer_ids = tokenizer(
        answer_text,
        return_tensors="pt",
        add_special_tokens=False
    ).input_ids.to(device)

    # Concatenate
    input_ids = torch.cat([prompt_ids, answer_ids], dim=1)

    prompt_len = prompt_ids.shape[1]
    answer_len = answer_ids.shape[1]

    # Structured mask (ONLY on answer)
    ans_mask = structured_mask(answer_len, device)
    full_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    full_mask[:, prompt_len:][0, ans_mask] = True

    # Mask token
    mask_id = model.config.mask_token_id
    noisy_ids = input_ids.clone()
    noisy_ids[full_mask] = mask_id

    # Forward
    outputs = model(noisy_ids)
    logits = outputs.logits

    # CE loss on masked positions
    loss = F.cross_entropy(
        logits[full_mask],
        input_ids[full_mask],
        reduction="mean"
    )

    return loss
