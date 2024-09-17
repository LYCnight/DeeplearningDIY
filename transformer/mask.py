import torch
import torch.nn as nn

# 模拟输入，假设 batch_size=1, n_heads=2, seq_len=4, d_k=3
batch_size = 1
n_heads = 2
seq_len = 4
d_k = 3

# 随机生成 query, key, value 张量
query = torch.randn(batch_size, n_heads, seq_len, d_k)
key = torch.randn(batch_size, n_heads, seq_len, d_k)
value = torch.randn(batch_size, n_heads, seq_len, d_k)

# 构建下三角矩阵 mask, 确保未来的 token 被屏蔽
mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
print(mask)
# tensor([[[[1., 0., 0., 0.],
#           [1., 1., 0., 0.],
#           [1., 1., 1., 0.],
#           [1., 1., 1., 1.]]]])

# mask = mask.expand(batch_size, n_heads, seq_len, seq_len)  # [batch_size, n_heads, seq_len, seq_len]
# print(mask)

# 初始化 ScaledDotProductAttention
class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.scale = 1.0 / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

    def forward(self, query, key, value, mask=None):
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)  # 屏蔽未来的 token
        attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_weights, value)
        return output, attn_weights

# 初始化并测试
attention = ScaledDotProductAttention(d_k)
output, attn_weights = attention(query, key, value, mask)

# 打印注意力权重和输出
print("Attention Weights:")
print(attn_weights)
print("Output:")
# print(output)
