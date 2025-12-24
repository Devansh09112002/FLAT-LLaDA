import torch

from llada_score_train import llada_score_train
from flat_contrastive_loss import get_contrastive_loss

def llada_flat_loss_train(model, tokenizer, prompt, forget, template,
                          mc_num=8, div="KL"):

    score_f = llada_score_train(model, tokenizer, prompt, forget, mc_num)
    score_e = llada_score_train(model, tokenizer, prompt, template, mc_num)
        # Clamp log-likelihoods to avoid exp explosion
    score_f = torch.clamp(score_f, min=-50.0, max=50.0)
    score_e = torch.clamp(score_e, min=-50.0, max=50.0)

    TEMP = 0.1  # start small

    score_f = score_f * TEMP
    score_e = score_e * TEMP

    loss = get_contrastive_loss(
        prob_sum_unlearn=score_f,
        prob_sum_good=score_e,
        div=div
    )

    return loss
