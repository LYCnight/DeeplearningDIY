'''
单头注意力的实现
'''

import torch
import torch.nn as nn
import math

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model):
        super(ScaledDotProductAttention, self).__init__()
        self.d_model = d_model
        self.scale = 1.0 / math.sqrt(d_model)

    def forward(self, query, key, value):
        # 计算注意力分数
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale  # [batch_size, seq_len, seq_len]
        # query                  [batch_size, seq_len, d_model]
        # key.transpose(-2, -1)  [batch_size, d_model, seq_len]
        
        # 计算注意力权重
        attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)  # [batch_size, seq_len, seq_len]
        
        # 计算加权和
        output = torch.matmul(attn_weights, value)  # [batch_size, seq_len, d_model]
        # attn_wrights          [batch_size, seq_len, seq_len]
        # value                 [batch_size, seq_len, d_model]
        
        # 
        return output, attn_weights

class SimpleSelfAttention(nn.Module):
    def __init__(self, d_model):
        super(SimpleSelfAttention, self).__init__()
        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention(d_model)

    def forward(self, x):
        # 生成查询、键和值
        query = self.query_linear(x)  # [batch_size, seq_len, d_model]
        key = self.key_linear(x)      # [batch_size, seq_len, d_model]
        value = self.value_linear(x)  # [batch_size, seq_len, d_model]
        
        # 计算自注意力
        output, attn_weights = self.attention(query, key, value)
        return output, attn_weights