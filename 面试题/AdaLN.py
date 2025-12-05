import torch
import torch.nn as nn

# ------------------------------
# 1. 定义 AdaLN
# ------------------------------
class AdaLN(nn.Module):
    def __init__(self, dim, cond_dim):
        super().__init__()
        self.fc = nn.Linear(cond_dim, dim * 2)
        self.eps = 1e-5

    def forward(self, x, cond):
        """
        x: (B, T, D)
        cond: (B, cond_dim)
        """
        # dim = 4
        import ipdb; ipdb.set_trace() 
        B, T, D = x.shape   # (2, 5, 4)
        gamma, beta = self.fc(cond).chunk(2, dim=-1)  # (B, D)
        # gamma # (2, 4)
        # beta: # (2, 4)

        # LayerNorm
        mean = x.mean(dim=-1, keepdim=True)  # (2, 5, 1)
        var = x.var(dim=-1, unbiased=False, keepdim=True)  # (2, 5, 1)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)   # (2, 5, 4)

        # Apply adaptive scale and shift
        out = x_norm * gamma.unsqueeze(1) + beta.unsqueeze(1)
        return out


# ------------------------------
# 2. 构造测试数据
# ------------------------------
B, T, D = 2, 5, 4
COND_DIM = 8

x = torch.randn(B, T, D)
cond = torch.randn(B, COND_DIM)

print("Input x:\n", x)
print("\nCondition vector:\n", cond)

# ------------------------------
# 3. 通过 AdaLN
# ------------------------------
adaln = AdaLN(dim=D, cond_dim=COND_DIM)
out = adaln(x, cond)

print("\nOutput after AdaLN:\n", out)

# ------------------------------
# 4. 对比统计量
# ------------------------------
print("\n--- Statistics ---")
print("Input mean (per batch):", x.mean(dim=(1,2)))
print("Output mean (per batch):", out.mean(dim=(1,2)))
print("Input std  (per batch):", x.std(dim=(1,2)))
print("Output std  (per batch):", out.std(dim=(1,2)))
