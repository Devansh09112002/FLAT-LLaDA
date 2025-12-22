# llada_flat_loss_train.py
from flat_contrastive_loss import get_contrastive_loss
from llada_denoise_loss import llada_denoise_loss

def llada_flat_loss_train(model, tokenizer, prompt, forget, template, div="KL"):
    loss_f = llada_denoise_loss(model, tokenizer, prompt, forget)
    loss_e = llada_denoise_loss(model, tokenizer, prompt, template)

    # IMPORTANT:
    # FLAT expects "score" where higher = better
    # denoise loss: lower = better
    score_f = -loss_f
    score_e = -loss_e

    loss = get_contrastive_loss(
        prob_sum_unlearn=score_f,
        prob_sum_good=score_e,
        div=div
    )

    return loss
