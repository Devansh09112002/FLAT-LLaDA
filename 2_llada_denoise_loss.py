import torch
import torch.nn.functional as F


def llada_denoise_loss(
    model,
    tokenizer,
    prompt_text: str,
    answer_text: str,
    mask_prob: float = 0.15,
):
    
    # print("Function signature:")
    # print(inspect.signature(llada_denoise_loss))
    
    """
    Compute RAW denoising cross-entropy loss for LLaDA.

    This is a TRAINABLE loss (no detach, no MC, no importance weighting).
    It measures how hard it is for the model to denoise masked answer tokens.

    Args:
        model: LLaDA model (train mode)
        tokenizer: corresponding tokenizer
        prompt_text: conditioning prompt (string)
        answer_text: target answer (string)
        mask_prob: probability of masking each answer token

    Returns:
        loss: scalar torch.Tensor with requires_grad=True
    """

    device = model.device

    # --------------------------------------------------
    # STEP A1.1 — Tokenize prompt and answer separately
    # --------------------------------------------------
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

    # --------------------------------------------------
    # STEP A1.2 — Concatenate prompt + answer
    # --------------------------------------------------
    input_ids = torch.cat([prompt_ids, answer_ids], dim=1)  # (1, L)

    # --------------------------------------------------
    # STEP A1.3 — Create mask ONLY over answer tokens
    # --------------------------------------------------
    seq_len = input_ids.shape[1]
    prompt_len = prompt_ids.shape[1]
    answer_len = answer_ids.shape[1]

    # Boolean mask over full sequence
    mask = torch.zeros((1, seq_len), dtype=torch.bool, device=device)

    # Randomly select masked positions in the ANSWER part only
    rand = torch.rand(answer_len, device=device)
    mask_positions = rand < mask_prob

    mask[:, prompt_len:][0, mask_positions] = True

    # Safety check: at least one token masked
    if mask.sum() == 0:
        mask[0, prompt_len] = True

    # --------------------------------------------------
    # STEP A1.4 — Replace masked tokens with [MASK]
    # --------------------------------------------------
    if hasattr(model.config, "mask_token_id"):
        mask_id = model.config.mask_token_id
    else:
        # Fallback: LLaDA default mask id
        mask_id = 126336

    noisy_input_ids = input_ids.clone()
    noisy_input_ids[mask] = mask_id

    # --------------------------------------------------
    # STEP A1.5 — Forward pass (NO torch.no_grad)
    # --------------------------------------------------
    outputs = model(noisy_input_ids)
    logits = outputs.logits  # (1, L, V)

    # --------------------------------------------------
    # STEP A1.6 — Compute denoising cross-entropy loss
    # --------------------------------------------------
    loss = F.cross_entropy(
        logits[mask],        # predictions at masked positions
        input_ids[mask],     # true tokens
        reduction="mean"
    )

    return loss
