import torch
from llada_loglikelihood_train import get_log_likelihood_train
# def llada_score_train(model, tokenizer, prompt, answer, mc_num=8):
#     device = model.device

#     prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
#     answer_ids = tokenizer(answer, return_tensors="pt").input_ids.to(device)

#     score = get_log_likelihood_train(
#         model,
#         prompt_ids,
#         answer_ids,
#         mc_num=mc_num
#     )
def llada_score_train(model, tokenizer, prompt_text, answer_text, mc_num=16):
    prompt = torch.tensor(tokenizer(prompt_text)['input_ids'], device=model.device)
    answer = torch.tensor(tokenizer(answer_text)['input_ids'], device=model.device)

    return get_log_likelihood_train(
        model,
        prompt,
        answer,
        mc_num=mc_num
    )


    return score
