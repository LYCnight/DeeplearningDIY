import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0  # 保证能整除

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # 生成 Q K V 的线性层
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # 多头拼接后再线性变换
        self.fc_out = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        B, L, _ = x.shape

        # 1) Linear projections
        Q = self.W_q(x)  # (B,L,d_model)
        K = self.W_k(x)
        V = self.W_v(x)

        # 2) reshape to multi-heads: (B, num_heads, L, head_dim)
        Q = Q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # 3) Scaled dot-product attention
        # attention scores = Q @ K^T
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)  # (B, heads, L, L)

        # 4) 加权求和
        out = torch.matmul(attn, V)  # (B, heads, L, head_dim)

        # 5) 拼接多头： (B, L, d_model)
        out = out.transpose(1, 2).contiguous().view(B, L, -1)

        # 6) 最后线性层
        out = self.fc_out(out)
        return out

