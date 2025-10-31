import torch

# CE = -log(e^qk / sum e^qk) = -qk + log(sum e^qk)

def cross_entropy_from_logits(logits, labels):
    """
    logits: (N, C) 未经过 softmax
    labels: (N,) 每项是 class index
    """
    # logsumexp：log(∑ exp(logits))
    lse = torch.logsumexp(logits, dim=1)          # (N,)
    '''
    # 手动计算 log(sum(exp()))
    exp_logits = torch.exp(logits)         # (N, C)
    sum_exp = torch.sum(exp_logits, dim=1) # (N,)
    lse = torch.log(sum_exp)               # (N,)
    '''

    # 取正确类别的 logit：z_y
    z_y = logits[torch.arange(logits.size(0)), labels]  # (N,)
    
    # CE = -z_y + logsumexp
    loss = -z_y + lse                           # (N,)

    return loss.mean()                            # 标量

if __name__ == "__main__":
    import torch
    import torch.nn.functional as F

    logits = torch.randn(4, 10)
    labels = torch.tensor([1, 3, 0, 5])

    loss_ref = F.cross_entropy(logits, labels)
    loss_my  = cross_entropy_from_logits(logits, labels)

    print(loss_ref.item(), loss_my.item())


