import torch

def get_contrastive_loss(prob_sum_unlearn, prob_sum_good, div='KL'):
    if div == 'KL':
        def activation(x): return -torch.mean(x)
        def conjugate(x): return -torch.mean(torch.exp(x - 1.))

    elif div == 'Total-Variation':
        def activation(x): return -torch.mean(torch.tanh(x) / 2.)
        def conjugate(x): return -torch.mean(torch.tanh(x) / 2.)

    else:
        raise NotImplementedError

    prob_reg = -prob_sum_good
    loss_regular = activation(prob_reg)

    prob_peer = -prob_sum_unlearn
    loss_peer = conjugate(prob_peer)

    loss = loss_regular - loss_peer
    return loss
