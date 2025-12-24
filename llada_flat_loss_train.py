# llada_flat_loss_train.py
from flat_contrastive_loss import get_contrastive_loss
from llada_denoise_loss import llada_denoise_loss

def llada_flat_loss_train(
    model,
    tokenizer,
    prompt,
    forget,
    template,
    mc_num=16,
    div="KL"
):
    score_f = llada_score_train(model, tokenizer, prompt, forget, mc_num)
    score_e = llada_score_train(model, tokenizer, prompt, template, mc_num)

    # ---- STABILIZATION ----
    TEMP = 0.2
    score_f = torch.clamp(score_f * TEMP, min=-20.0, max=20.0)
    score_e = torch.clamp(score_e * TEMP, min=-20.0, max=20.0)

    loss = get_contrastive_loss(
        prob_sum_unlearn=score_f,
        prob_sum_good=score_e,
        div=div
    )

    return loss

