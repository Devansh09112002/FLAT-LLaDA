import torch
from get_log_likelihood import get_log_likelihood

def llada_score(model, tokenizer, prompt_text, answer_text, mc_num=32):
    device = model.device

    prompt_ids = torch.tensor(
        tokenizer(prompt_text)['input_ids'],
        device=device
    )

    answer_ids = torch.tensor(
        tokenizer(answer_text)['input_ids'],
        device=device
    )

    score = get_log_likelihood(
        model,
        prompt_ids,
        answer_ids,
        mc_num=mc_num
    )

    # 🔑 NORMALIZATION (critical)
    num_tokens = len(answer_ids)
    score = score / num_tokens

    return score


