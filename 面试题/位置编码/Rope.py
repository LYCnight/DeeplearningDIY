import torch
import math

def build_rope_cache(seq_len, dim, device):
    """生成 cos/sin 矩阵，用于每个位置的旋转
    最终输出:
        emb_cos, emb_sin: (1, seq_len, dim)
    """
    # -----------------------------------------------
    # 1️⃣ 每两个维度一组，因此 dim 必须为偶数
    #    half_dim = d/2，表示有多少个“旋转平面”
    # -----------------------------------------------
    half_dim = dim // 2  # scalar, e.g. dim=512 -> half_dim=256

    # -----------------------------------------------
    # 2️⃣ 构建频率序列
    #    freq_seq = [0, 1, 2, ..., half_dim-1]
    #    shape: (half_dim,)
    # -----------------------------------------------
    freq_seq = torch.arange(0, half_dim, device=device, dtype=torch.float32)
    # freq_seq: (256,)

    # -----------------------------------------------
    # 3️⃣ 计算不同维度的角频率 inv_freq
    #    按论文公式: 10000^{-2i/d}
    #    这里是 10000^{-(i/half_dim)}
    #    shape: (half_dim,)
    # -----------------------------------------------
    inv_freq = 1.0 / (10000 ** (freq_seq / half_dim))
    # inv_freq: (256,)
    # 表示每个维度的“旋转速度”不同，高维旋转慢，低维旋转快

    # -----------------------------------------------
    # 4️⃣ 生成位置索引序列
    #    t = [0, 1, 2, ..., seq_len-1]
    #    shape: (seq_len,)
    # -----------------------------------------------
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    # t: (L,)

    # -----------------------------------------------
    # 5️⃣ 外积计算 freqs = t * inv_freq
    #    einsum("i,j->ij") 表示: freqs[i,j] = t[i] * inv_freq[j]
    #    shape: (seq_len, half_dim)
    # -----------------------------------------------
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    # freqs: (L, half_dim)
    # 对应每个位置 t_i、每个频率维度 j 的角度 (mθ)

    # -----------------------------------------------
    # 6️⃣ 计算 cos/sin 编码
    #    cos/sin(freqs) → (seq_len, half_dim)
    # -----------------------------------------------
    emb_cos = torch.cos(freqs)  # (L, half_dim)
    emb_sin = torch.sin(freqs)  # (L, half_dim)

    # -----------------------------------------------
    # 7️⃣ 将每个半维度重复两次 (因为 q0/q1 成对使用)
    #    repeat_interleave(..., 2, dim=-1) 把每个列复制一份
    #    例如：原本 (L, 256) → (L, 512)
    # -----------------------------------------------
    emb_cos = torch.repeat_interleave(emb_cos, 2, dim=-1).unsqueeze(0)
    emb_sin = torch.repeat_interleave(emb_sin, 2, dim=-1).unsqueeze(0)
    # emb_cos, emb_sin: (1, seq_len, dim)
    # 加 unsqueeze(0) 是为了在 batch 维度上可以广播

    # -----------------------------------------------
    # ✅ 最终输出:
    #    emb_cos: (1, seq_len, dim)
    #    emb_sin: (1, seq_len, dim)
    # -----------------------------------------------
    return emb_cos, emb_sin


def apply_rope(x, cos, sin):
    """
    x:   (batch, seq_len, dim)
    cos: (1, seq_len, dim)
    sin: (1, seq_len, dim)
    说明：cos/sin 的第一个 batch 维是 1，会在广播时自动扩展
    """
    # -------------------------------
    # 1️⃣ 拆分偶数维和奇数维
    # -------------------------------
    x1 = x[..., 0::2]   # 取偶数下标维度 -> (batch, seq_len, dim/2)
    x2 = x[..., 1::2]   # 取奇数下标维度 -> (batch, seq_len, dim/2)

    # cos[..., 0::2] 也是 (1, seq_len, dim/2)
    # sin[..., 0::2] 也是 (1, seq_len, dim/2)
    # 它们会自动广播到 batch 维度上

    # -------------------------------
    # 2️⃣ 执行二维旋转
    # 其实就是：
    # q0' = q0 * cos - q1 * sin
    # q1' = q0 * sin + q1 * cos
    # -------------------------------
    x_rot  = x1 * cos[..., 0::2] - x2 * sin[..., 0::2]   # (batch, seq_len, dim/2)
    x_rot2 = x1 * sin[..., 0::2] + x2 * cos[..., 0::2]   # (batch, seq_len, dim/2)

    # -------------------------------
    # 3️⃣ 拼回原来的维度顺序
    # stack 在最后一维加个维度 -> (batch, seq_len, dim/2, 2)
    # 再 reshape 回 (batch, seq_len, dim)
    # -------------------------------
    x_out = torch.stack([x_rot, x_rot2], dim=-1).reshape_as(x)
    # 过程：
    # [x_rot, x_rot2]  →  shape = (2, batch, seq_len, dim/2)
    # stack(dim=-1)    →  shape = (batch, seq_len, dim/2, 2)
    # reshape_as(x)    →  shape = (batch, seq_len, dim)
    
    return x_out



class RoPEAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.W_q = torch.nn.Linear(d_model, d_model)
        self.W_k = torch.nn.Linear(d_model, d_model)
        self.W_v = torch.nn.Linear(d_model, d_model)
        self.fc_out = torch.nn.Linear(d_model, d_model)

    def forward(self, x):
        B, L, D = x.shape
        device = x.device

        # Q, K, V projection
        Q = self.W_q(x).view(B, L, self.num_heads, self.head_dim)
        K = self.W_k(x).view(B, L, self.num_heads, self.head_dim)
        V = self.W_v(x).view(B, L, self.num_heads, self.head_dim)

        # 生成旋转矩阵
        cos, sin = build_rope_cache(L, self.head_dim, device)

        # 应用 RoPE
        Q = apply_rope(Q, cos, sin)
        K = apply_rope(K, cos, sin)

        # 注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)

        # 拼回原维度
        out = out.reshape(B, L, -1)
        return self.fc_out(out)



