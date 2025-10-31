import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=4, alpha=1.0, bias=True):
        super().__init__()
        
        # 原始权重 W: (in, out)
        self.weight = nn.Parameter(torch.randn(in_features, out_features) * 0.02)
        self.weight.requires_grad = False  # 冻结
        
        # LoRA 矩阵
        self.A = nn.linear(in_features, out_features)
        self.B = nn.linear(out_features, in_features)
        
        self.alpha = alpha
        self.r = r
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

    def forward(self, x):
        # x: (B, in)
        x_origin = x @ self.weight                 # (B, out)
        if self.bias is not None:
            x_origin = x_origin + self.bias        # 加 bias
        
        x_lora = self.B(self.A(x))                 # (B, out)
        return x_origin + (self.alpha / self.r) * x_lora



