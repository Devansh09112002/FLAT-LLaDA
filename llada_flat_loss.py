import torch
from llada_score import llada_score
from flat_contrastive_loss import get_contrastive_loss

def llada_flat_loss(model, tokenizer, prompt, forget, template,
                    mc_num=16, div='KL'):

    score_f = torch.tensor(
        llada_score(model, tokenizer, prompt, forget, mc_num),
        device=model.device
    )

    score_e = torch.tensor(
        llada_score(model, tokenizer, prompt, template, mc_num),
        device=model.device
    )

    loss = get_contrastive_loss(
        prob_sum_unlearn=score_f,
        prob_sum_good=score_e,
        div=div
    )

    return loss

