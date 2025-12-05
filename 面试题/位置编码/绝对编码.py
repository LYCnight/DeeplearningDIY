import torch
import torch.nn as nn
import math

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        # (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)  # (max_len,1)

        # 角速度
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math .log(10000.0) / d_model)
        )  # (d_model/2,)

        # or 简写版
        # div_term = (10000 ** (-torch.arange(0, d_model, 2).float() / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度   # (max_len, d_model/2)
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度   # (max_len, d_model/2)

        # 注册为 buffer，不作为训练参数
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]
    



