import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Embeddings(nn.Module):
    '''
    - batch_size = 批大小，seq_len = 序列长度，vocab_size = 词汇表大小，d_model = 模型维度
    - 输入 x 的形状是 [batch_size, seq_len]。
    - nn.Embedding 的形状是 [vocab_size, d_model]。
    - self.embedding(x) 的形状是 [batch_size, seq_len, d_model]。
    -- return size = [batch_size, seq_len, d_model]
    '''
    def __init__(self, vocab_size, d_model):
        super(Embeddings, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)  # [vocab_size, d_model]，映射词表
        self.d_model = d_model
    
    def forward(self, x):
        # x 的形状是 [batch_size, seq_len], (如果batch_size 1，则 x 的形状是 [seq_len])
        # 返回的形状是 [batch_size, seq_len, d_model], (如果batch_size = 1，则返回的形状是 [seq_len, d_model])
        '''
        - 为什么乘以 sqrt(d_model)：目的是通过缩放嵌入向量的数值范围，避免点积注意力中的数值过大，从而保持训练过程的稳定性。
        - return_sie: [batch_size, seq_len, d_model]
        '''
        # 将 embedding 乘以 sqrt(d_model)，确保缩放
        return self.embedding(x) * math.sqrt(self.d_model)
    

class PositionalEncoding(nn.Module):
    '''
    - batch_size = 批大小，seq_len = 序列长度，max_len = 位置编码所支持的最大序列长度, vocab_size = 词汇表大小，d_model = 模型维度
    - 输入 x 的形状是 [batch_size, seq_len, d_model]。
    -- return size = [batch_size, seq_len, d_model]
    '''
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # 创建一个 max_len × d_model 的位置编码矩阵
        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # [max_len, 1] (列向量)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # [d_model/2] (向量)    
        # arange(0, d_model, 2) 左闭右开，[0, 2, 4, ..., d_model-2]
        
        # 按照公式计算 sin 和 cos 的位置编码
        pe[:, 0::2] = torch.sin(position * div_term)    # [max_len, d_model / 2]
        pe[:, 1::2] = torch.cos(position * div_term)    # [max_len, d_model / 2]
        
        # 将 pe 扩展一个 batch 维度
        # 这样做的目的是为批量处理添加一个“batch”维度，通常在深度学习模型中，我们需要一个批量维度来处理多个序列。
        pe = pe.unsqueeze(0)    # [1, max_len, d_model]
        
        # self.pe 是通过 self.register_buffer('pe', pe) 定义的。
        # 用于在模型中注册一个不可训练的参数（如位置编码）
        self.register_buffer('pe', pe)  
    
    def forward(self, x):
        # 在输入上加上位置编码
        # x 的形状通常是 (batch_size, seq_len, d_model)
        # print(self.pe.size())    # [1, max_len, d_model]
        # print(self.pe[:, :x.size(1), :].size())    # [1, max_len, d_model]
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]  # 切片语法  
        # [batch_size, seq_len, d_model] + [1, seq_len, d_model] 被广播为 [batch_size, seq_len, d_model] + [batch_size, seq_len, d_model]
        return x    # [batch_size, seq_len, d_model]
    
    
class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.scale = 1.0 / math.sqrt(d_k)

    def forward(self, query, key, value, mask=None):
        '''
        query, key, value 的形状都是 [batch_size, n_heads, seq_len, d_k]
        mask 的形状是 [batch_size, n_heads, seq_len, seq_len]，或者可以广播为这个形状
        output 的形状是 [batch_size, n_heads, seq_len, d_k]
        attn_weights 的形状是 [batch_size, n_heads, seq_len, seq_len]
        '''
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale  # [batch_size, n_heads, seq_len, seq_len]

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)  # 将被 mask 掉的部分设为负无穷大
            # Tensor.masked_fill(mask, value)
            # mask：一个与目标张量形状相同的布尔张量（或可以广播成相同形状）。mask == 0 表示：  mask == 0 的位置就是需要被替换的位置
            # value：替换的值。在被屏蔽的位置，原始张量的值将被替换为 value

        attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)  # [batch_size, n_heads, seq_len, seq_len]
        output = torch.matmul(attn_weights, value)  # [batch_size, n_heads, seq_len, d_k]
        return output, attn_weights
    

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0, "d_model必须能够被n_heads整除"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # 每个头的维度

        # 线性层将 d_model 维度映射到 n_heads 个 d_k 维度
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

        # 最后的线性层用于恢复到原始 d_model 维度
        self.out_linear = nn.Linear(d_model, d_model)

        # 自注意力层
        self.attention = ScaledDotProductAttention(self.d_k)

    def forward(self, x, mask=None):  
        batch_size, seq_len, _ = x.size()   # [batch_size, seq_len, d_model]

        # 线性变换并重塑为多头格式: [batch_size, seq_len, d_model] -> [batch_size, n_heads, seq_len, d_k]
        query = self.q_linear(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)  # [batch_size, n_heads, seq_len, d_k]
        key = self.k_linear(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)    # [batch_size, n_heads, seq_len, d_k]
        value = self.v_linear(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)  # [batch_size, n_heads, seq_len, d_k]

        # 计算多头注意力: [batch_size, n_heads, seq_len, d_k] （并且将 mask 传递给 ScaledDotProductAttention）
        attn_output, attn_weights = self.attention(query, key, value, mask)
        # attn_output    # [batch_size, n_heads, seq_len, d_k]
        # attn_weights   # [batch_size, n_heads, seq_len, seq_len]

        # 将多头注意力的输出拼接回原始形状: [batch_size, seq_len, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        # attn_output.transpose(1, 2) # [batch_size, seq_len, n_heads, d_k]

        # 通过线性层恢复到 d_model 维度, 线性层帮助模型更好地整合多个头的注意力输出。
        output = self.out_linear(attn_output)   
        # output :       [batch_size, seq_len, d_model]
        # attn_weights : [batch_size, n_heads, seq_len, seq_len]
        return output, attn_weights
    
    def cross_attention(self, q, k, v, mask=None):  
        batch_size, seq_len, _ = q.size()   # [batch_size, seq_len, d_model]

        # 线性变换并重塑为多头格式: [batch_size, seq_len, d_model] -> [batch_size, n_heads, seq_len, d_k]
        query = self.q_linear(q).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)  # [batch_size, n_heads, seq_len, d_k]
        key = self.k_linear(k).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)    # [batch_size, n_heads, seq_len, d_k]
        value = self.v_linear(v).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)  # [batch_size, n_heads, seq_len, d_k]

        # 计算多头注意力: [batch_size, n_heads, seq_len, d_k] （并且将 mask 传递给 ScaledDotProductAttention）
        attn_output, attn_weights = self.attention(query, key, value, mask)
        # attn_output    # [batch_size, n_heads, seq_len, d_k]
        # attn_weights   # [batch_size, n_heads, seq_len, seq_len]

        # 将多头注意力的输出拼接回原始形状: [batch_size, seq_len, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        # attn_output.transpose(1, 2) # [batch_size, seq_len, n_heads, d_k]

        # 通过线性层恢复到 d_model 维度, 线性层帮助模型更好地整合多个头的注意力输出。
        output = self.out_linear(attn_output)   
        # output :       [batch_size, seq_len, d_model]
        # attn_weights : [batch_size, n_heads, seq_len, seq_len]
        return output, attn_weights

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        """
        Layer Normalization 层
        - d_model: 模型维度
        - eps: 防止除零的微小值
        """
        super(LayerNorm, self).__init__()
        self.scale = nn.Parameter(torch.ones(d_model))  # 缩放参数 γ，初始化为 1    [d_model]
        self.shift = nn.Parameter(torch.zeros(d_model)) # 平移参数 β，初始化为 0    [d_model]
        self.eps = eps

    def forward(self, x):
        '''
        return : [batch_size, seq_len, d_model]
        '''
        # x 的形状 [batch_size, seq_len, d_model]
        mean = x.mean(dim=-1, keepdim=True)             # 计算最后一维的均值    [batch_size, seq_len, 1]
        std = x.std(dim=-1, keepdim=True)               # 计算最后一维的标准差  [batch_size, seq_len, 1]
        # 进行标准化并应用缩放和平移
        return self.scale * (x - mean) / (std + self.eps) + self.shift

class ResidualConnection(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        """
        残差连接和层归一化
        - d_model: 模型维度
        - dropout: dropout 率，防止过拟合
        """
        super(ResidualConnection, self).__init__()
        self.layer_norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_output):
        """
        - x: 输入注意力层之前的张量        [batch_size, seq_len, d_model]
        - attn_output: 注意力层输出张量   [batch_size, seq_len, d_model]
        """
        return self.layer_norm(x + self.dropout(attn_output))  # 残差连接 + 层归一化

    
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        前馈网络层
        - d_model: 模型维度
        - d_ff: 前馈网络的隐藏层大小，通常设置为 4 * d_model
        - dropout: dropout 率，防止过拟合
        """
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)  # 第一层线性变换
        self.linear2 = nn.Linear(d_ff, d_model)  # 第二层线性变换
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        - x: 输入张量 [batch_size, seq_len, d_model]
        - 返回：前馈网络的输出 [batch_size, seq_len, d_model]
        """
        # 第一层线性变换后应用 ReLU 激活函数，然后使用 dropout，再通过第二层线性变换
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

# 单层 Encoderlayer
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        """
        编码器层
        - d_model: 模型维度
        - n_heads: 多头注意力的头数
        - d_ff: 前馈网络的隐藏层大小
        - dropout: dropout 率
        """
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads)  # 多头自注意力层
        self.feed_forward = FeedForward(d_model, d_ff, dropout)     # 前馈网络
        self.residual1 = ResidualConnection(d_model, dropout)       # 第一个残差连接
        self.residual2 = ResidualConnection(d_model, dropout)       # 第二个残差连接

    def forward(self, x):
        """
        - x: 输入张量 [batch_size, seq_len, d_model]
        - 返回：编码器层的输出 [batch_size, seq_len, d_model]
        """
        # 1. 通过自注意力层和残差连接
        attn_output, attn_weights = self.self_attention(x)
        x = self.residual1(x, attn_output)  # [batch_size, seq_len, d_model]

        # 2. 通过前馈网络和残差连接
        ff_output = self.feed_forward(x)
        x = self.residual2(x, ff_output)  # [batch_size, seq_len, d_model]

        return x

# 完整 Encoder 
class Encoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, num_layers, dropout=0.1):
        """
        整体编码器
        - d_model: 模型维度
        - n_heads: 多头注意力的头数
        - d_ff: 前馈网络隐藏层大小
        - num_layers: 编码器层的堆叠数
        - dropout: dropout 率
        """
        super(Encoder, self).__init__()
        
        
        # 堆叠多个 EncoderLayer
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)])

    def forward(self, x):
        """
        - x: 输入张量 [batch_size, seq_len]
        - 返回: 编码器的输出 [batch_size, seq_len, d_model]
        """        
        # 通过多个 EncoderLayer 处理
        for layer in self.layers:
            x = layer(x)
        
        return x
 

# 单层 DecoderLayer
class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        """
        解码器层
        - d_model: 模型维度
        - n_heads: 多头注意力的头数
        - d_ff: 前馈网络的隐藏层大小
        - dropout: dropout 率
        """
        super(DecoderLayer, self).__init__()
        # 掩码自注意力（Masked Multi-Head Attention）
        self.masked_self_attention = MultiHeadAttention(d_model, n_heads)
        
        # Encoder-Decoder Attention（跨注意力）
        self.enc_dec_attention = MultiHeadAttention(d_model, n_heads)
        
        # 前馈网络
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        # 残差连接和层归一化
        self.residual1 = ResidualConnection(d_model, dropout)
        self.residual2 = ResidualConnection(d_model, dropout)
        self.residual3 = ResidualConnection(d_model, dropout)
    
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        """
        - x: 解码器输入 [batch_size, tgt_seq_len, d_model]
        - enc_output: 编码器的输出 [batch_size, src_seq_len, d_model]
        - src_mask: 用于编码器的输入 mask [batch_size, 1, 1, src_seq_len] （可选）
        - tgt_mask: 用于解码器输入的掩码 [batch_size, 1, tgt_seq_len, tgt_seq_len]
        - 返回：解码器层的输出 [batch_size, tgt_seq_len, d_model]
        """
        # 1. 掩码自注意力（Masked Multi-Head Attention）
        attn_output1, _ = self.masked_self_attention(x, tgt_mask)
        x = self.residual1(x, attn_output1)  # [batch_size, tgt_seq_len, d_model]
        
        # 2. Encoder-Decoder Attention（跨注意力）
        attn_output2, _ = self.enc_dec_attention.cross_attention(q=x, k=enc_output, v=enc_output, mask=src_mask)
        x = self.residual2(x, attn_output2)  # [batch_size, tgt_seq_len, d_model]
        
        # 3. 前馈网络
        ff_output = self.feed_forward(x)
        x = self.residual3(x, ff_output)  # [batch_size, tgt_seq_len, d_model]
        
        return x

# 完整 Decoder
class Decoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, n_layers, dropout=0.1):
        """
        解码器
        - d_model: 模型维度
        - n_heads: 多头注意力的头数
        - d_ff: 前馈网络的隐藏层大小
        - n_layers: 解码器层数
        - dropout: dropout 率
        """
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])

    def forward(self, tgt_seq, enc_output, src_mask=None, tgt_mask=None):
        """
        - tgt_seq: 目标序列输入 [batch_size, tgt_seq_len, d_model]
        - enc_output: 编码器的输出 [batch_size, src_seq_len, d_model]
        - src_mask: 用于编码器的输入 mask [batch_size, 1, 1, src_seq_len] （可选）
        - tgt_mask: 用于解码器输入的掩码 [batch_size, 1, tgt_seq_len, tgt_seq_len]
        - 返回：解码器的输出 [batch_size, tgt_seq_len, d_model]
        """
        x = tgt_seq

        # 逐层处理
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)

        return x

# Transformer
class Transformer(nn.Module):
    def __init__(self, encoder, decoder):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, src_mask, tgt_mask, memory_mask):
        memory = self.encoder(src, src_mask)
        output = self.decoder(tgt, memory, tgt_mask, memory_mask)
        return output


if __name__ == '__main__':

    # 超参数
    vocab_size = 10000  # 词汇表大小
    d_model = 512       # 模型维度
    seq_len = 10        # 序列长度, 如果为None, 则计算并使用 batch 中最长的序列长度
    max_len = 5000      # 位置编码所支持的最大序列长度
    batch_size = 3      # 批次大小
    n_head = 8          # 多头注意力的头数
    dropout = 0.1       # dropout 率
    encoder_layer_nums = 6  # 编码器层数量
    decoder_layer_nums = 6  # 解码器层数量
    d_ff = 2024;  # 前馈网络的隐藏层大小
    
    # 词表
    from utils import vocab

    # 模型实例化
    embedding_layer = Embeddings(vocab_size, d_model)
    positional_encoding_layer = PositionalEncoding(d_model, max_len)
    encoder = Encoder(d_model, n_head, d_ff, encoder_layer_nums, dropout)
    decoder = Decoder(d_model, n_head, d_ff, decoder_layer_nums, dropout)  
    linear = nn.Linear(d_model, vocab_size)
    softmax = nn.Softmax(dim=-1)

    # 1. Encoder -------------------------------
    # 原始字符输入
    encoder_sequences = ["today hello world", "hello world I love U and She", "我 爱 你 !"]
    
    # 在这里写一个函数,转为 encoder_original_input
    from utils import prepare_encoder_input
    encoder_original_input, seq_len = prepare_encoder_input(encoder_sequences, vocab, seq_len)    # [batch_size, seq_len]
    print(encoder_original_input)
    # seq_len 其实是 encoder_sequences 中最长的序列长度,是计算得到的
    
    # 编码器输入(未经过 Embedding 和 Positional Encoding)
    # encoder_original_input = torch.randint(0, vocab_size, (batch_size, seq_len))     # [batch_size, seq_len]
    # Embedding + Positional Encoding
    embedded_enc = embedding_layer(encoder_original_input)
    encoder_input = positional_encoding_layer(embedded_enc)
    print("encoder_input:")
    print(encoder_input.size())  # [batch_size, seq_len, d_model]
    # encode
    encoder_output = encoder(encoder_input)
    print("Encoder Output:")
    print(encoder_output.size())  # [batch_size, seq_len, d_model]
    
    
    # 2. Decoder -------------------------------
    # 解码器输入(未经过 Embedding 和 Positional Encoding)
    decoder_original_input = torch.randint(0, vocab_size, (batch_size, seq_len))  # [batch_size, seq_len]
    # embedding & positional encoding
    embedded_dec = embedding_layer(decoder_original_input)
    decoder_input = positional_encoding_layer(embedded_dec)
    print("decoder_input:")
    print(decoder_input.size())  # [batch_size, seq_len, d_model]
    # 解码器输入掩码
    # 防止解码器在预测下一个词时看到当前词的未来词。
    tgt_mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len], 运算时会被广播成 [batch_size, n_head, seq_len, seq_len]
    # 编码器输入掩码
    # src_mask 是为了屏蔽 <pad> 或者填充标记（通常是 0），以防止这些填充位置对注意力计算产生影响。
    # 这种方式仅屏蔽填充标记，允许模型处理 <Unknown>，并学习如何对其进行预测或推理
    src_mask = (encoder_original_input != vocab["<PAD>"]).unsqueeze(1).unsqueeze(2)   # [batch_size, 1, 1, seq_len], 运算时会被广播成 [batch_size, n_head, seq_len, seq_len]
    # decode
    decoder_output = decoder(decoder_input, encoder_output, src_mask=src_mask, tgt_mask=tgt_mask)
    print("Decoder Output:")
    print(decoder_output.size())  # [batch_size, seq_len, d_model]
    
    # 3. predict -------------------------------
    linear_output = linear(decoder_output)
    softmax_output = softmax(linear_output)
    
    from utils import predict_words
    #  定义反向词汇表 (for word prediction)
    inverse_vocab = {index: word for word, index in vocab.items()}
    predicted_words:list[list[str]] = predict_words(softmax_output, inverse_vocab, seq_len)    # [batch_size, seq_len]  

    # 打印预测的词汇
    for i, sentence in enumerate(predicted_words):
        print(f"Sentence {i+1}: {' '.join(sentence)}")
