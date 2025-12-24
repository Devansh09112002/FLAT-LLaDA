# llada_loglikelihood_train.py
import torch
import torch.nn.functional as F


def forward_process(batch, prompt_index, mask_id):
    """
    IDENTICAL to original implementation
    """
    b, l = batch.shape
    target_len = (l - prompt_index.sum()).item()

    k = torch.randint(1, target_len + 1, (), device=batch.device)
    x = torch.round(
        torch.linspace(float(k), k + (b - 1) * (target_len / b),
                       steps=b, device=batch.device)
    ).long()
    x = ((x - 1) % target_len) + 1

    indices = torch.arange(target_len, device=batch.device).repeat(b, 1)
    is_mask = indices < x.unsqueeze(1)
    for i in range(b):
        is_mask[i] = is_mask[i][torch.randperm(target_len)]

    is_mask = torch.cat(
        (
            torch.zeros(b, prompt_index.sum(), dtype=torch.bool, device=batch.device),
            is_mask
        ),
        dim=1
    )

    noisy_batch = torch.where(is_mask, mask_id, batch)
    p_mask = (x / target_len).unsqueeze(1).repeat(1, l)

    return noisy_batch, p_mask


def get_logits(model, batch, prompt_index, cfg_scale, mask_id):
    """
    IDENTICAL to original implementation
    """
    if cfg_scale > 0.:
        prompt_index = prompt_index.unsqueeze(0).repeat(batch.shape[0], 1)
        un_batch = batch.clone()
        un_batch[prompt_index] = mask_id
        batch = torch.cat([batch, un_batch])

    logits = model(batch).logits

    if cfg_scale > 0.:
        logits, un_logits = torch.chunk(logits, 2, dim=0)
        logits = un_logits + (cfg_scale + 1) * (logits - un_logits)

    return logits


def get_log_likelihood_train(
    model,
    prompt,
    answer,
    mc_num=16,
    batch_size=4,
    cfg_scale=0.,
    mask_id=126336,
):
    """
    TRAINABLE version of get_log_likelihood

    Returns:
        scalar tensor (requires_grad=True)
    """

    device = model.device
    seq = torch.cat([prompt, answer])[None, :]
    seq = seq.repeat(batch_size, 1).to(device)

    prompt_index = torch.arange(seq.shape[1], device=device) < len(prompt)

    losses = []

    for _ in range(mc_num // batch_size):
        perturbed_seq, p_mask = forward_process(seq, prompt_index, mask_id)
        mask_index = perturbed_seq == mask_id

        logits = get_logits(model, perturbed_seq, prompt_index, cfg_scale, mask_id)

        # IMPORTANT: reduction='none' to preserve token-level loss
        token_loss = F.cross_entropy(
            logits[mask_index],
            seq[mask_index],
            reduction='none'
        )

        # importance weighting
        token_loss = token_loss / p_mask[mask_index]

        # sum over tokens, average over batch
        loss = token_loss.sum() / batch_size

        losses.append(loss)

    # Monte Carlo average — KEEP as tensor
    mean_loss = torch.stack(losses).mean()

    # log-likelihood = negative reconstruction loss
    return -mean_loss
