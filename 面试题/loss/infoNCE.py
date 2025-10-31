import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_info_nce(query: torch.Tensor, key: torch.Tensor, tau=0.07):
    """
    query: (N, D)
    key: (N, D)
    InfoNCE = -log( exp(q·k_pos / τ) / sum_j exp(q·k_j / τ ) )
    """
    # 计算相似度 (N, N)
    logits = torch.matmul(query, key.t()) / tau

    # 正样本是对角位置，所以 label = [0,1,2,...,N-1]
    N = query.size(0)
    labels = torch.arange(N).long()  # shape (N,)

    # CrossEntropy 内部会做 softmax，不需要手动 exp
    loss = F.cross_entropy(logits, labels)
    return loss


