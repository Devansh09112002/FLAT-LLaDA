import torch
import random


def structured_mask(answer_len, device):
    """
    Returns a boolean mask over answer tokens.
    """
    r = random.random()

    mask = torch.zeros(answer_len, dtype=torch.bool, device=device)

    # 40% contiguous span
    if r < 0.40:
        span_frac = random.uniform(0.3, 0.6)
        span_len = max(1, int(span_frac * answer_len))
        start = random.randint(0, answer_len - span_len)
        mask[start:start + span_len] = True

    # 25% clause-like chunk masking
    elif r < 0.65:
        num_chunks = random.randint(1, 3)
        for _ in range(num_chunks):
            span_len = random.randint(
                max(1, answer_len // 8),
                max(2, answer_len // 3)
            )
            start = random.randint(0, answer_len - span_len)
            mask[start:start + span_len] = True

    # 15% prefix-heavy masking
    elif r < 0.80:
        keep = max(1, int(0.1 * answer_len))
        mask[keep:] = True

    # 20% random token masking
    else:
        prob = 0.15
        mask = torch.rand(answer_len, device=device) < prob

    # Safety: ensure something is masked
    if not mask.any():
        mask[random.randint(0, answer_len - 1)] = True

    return mask
