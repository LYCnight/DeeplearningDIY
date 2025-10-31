class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, use_cache=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.use_cache = use_cache

    def forward(self, hidden_states, attention_mask=None, layer_past=None, use_cache=True):
        """
        hidden_states: [batch, seq_len, hidden_dim]
        layer_past: (past_key, past_value)
            past_key: [batch, num_heads, past_seq_len, head_dim]
            past_value: same shape
        """
        # 1) 线性变换得到 Q,K,V
        query = self.q_proj(hidden_states)     # [B, L, H]
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        # 2) reshape 成多头格式
        B, L, _ = query.size()
        query = query.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)   # [B, num_heads, L, head_dim]
        key   = key.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # =============== ✅ ✅ KV-CACHE 核心逻辑 =================
        if layer_past is not None:
            past_key, past_value = layer_past   # 已缓存历史 Key/Value

            # 把过去的 KV 拼接在前面
            # 注意：dim = -2 表示拼接 seq_len 维度
            key   = torch.cat([past_key, key], dim=-2)     # [B, num_heads, past+L, head_dim]
            value = torch.cat([past_value, value], dim=-2)

        # 如果需要缓存，返回这一层新的 KV
        if use_cache:
            present = (key, value)
        else:
            present = None
        # ========================================================

        # 3) 计算注意力 scores = QKᵀ / sqrt(d)
        attn_scores = torch.matmul(query, key.transpose(-1, -2)) / (self.head_dim ** 0.5)
        # attention_mask: 下三角矩阵，防止看到未来 token
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        attn_weights = torch.softmax(attn_scores, dim=-1)

        # 4) 得到输出
        attn_output = torch.matmul(attn_weights, value)   # [B, num_heads, L, head_dim]

        # 5) 合并 heads
        attn_output = attn_output.transpose(1, 2).reshape(B, L, self.hidden_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output, present



if __name__ == '__main__':
    past = None
    for step in range(max_len):
        out, present = transformer(input_token, past_key_value=past)
        past = present    # 缓存更新
        input_token = sample_next_token(out)
