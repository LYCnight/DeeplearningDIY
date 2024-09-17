# 假设我们有两个词汇表：中文和英文的词汇表
chinese_vocab = {
    '<PAD>': 0,
    '<START>': 1,
    '<END>': 2,
    '<Unknown>': 3,
    '你好': 4,
    '我': 5,
    '爱': 6,
    '编程': 7,
    '她': 8,
    '今天': 9,
    '很': 10,
    '开心': 11,
    '世界': 12,
    '和平': 13
}

english_vocab = {
    '<PAD>': 0,
    '<START>': 1,
    '<END>': 2,
    '<Unknown>': 3,
    'hello': 4,
    'I': 5,
    'love': 6,
    'programming': 7,
    'She': 8,
    'is': 9,
    'happy': 10,
    'today': 11,
    'world': 12,
    'peace': 13
}

from utils import prepare_encoder_input

def prepare_data_for_training(training_data, chinese_vocab, english_vocab):
    '''
    该函数用于处理机器翻译任务中的训练数据，将中文输入序列和英文目标序列转换为模型可以使用的张量格式。
    
    参数：
    - training_data: 训练数据，列表类型，其中每个元素是一个元组，包含 (中文句子, 英文句子)。
    - chinese_vocab: 中文词汇表，字典类型，键为中文词汇，值为对应的索引编号。
    - english_vocab: 英文词汇表，字典类型，键为英文词汇，值为对应的索引编号。
    
    返回：
    - encoder_inputs: 编码器的输入序列张量，中文句子转换成的索引序列，形状为 [batch_size, seq_len]。
    - decoder_inputs: 解码器的输入序列张量，英文句子转换成的索引序列，带有 <START> 和 <END> 标记，形状为 [batch_size, seq_len]。
    - chinese_seq_len: 中文序列的长度
    - english_seq_len: 英文序列的长度
    '''
    # 分别处理中文和英文句子
    chinese_sentences, english_sentences = zip(*training_data)
    # print(*training_data)   # ('你 好', 'hello') ('我 爱 编程', 'I love programming') ('她 今天 很 开心', 'She is happy today') ('世界 和平', 'world peace')
    # print(chinese_sentences)   # ('你 好', '我 爱 编程', '她 今天 很 开心', '世界 和平')

    # 准备编码器（中文）的输入
    encoder_inputs, chinese_seq_len = prepare_encoder_input(chinese_sentences, chinese_vocab)

    # 准备解码器（英文）的输入（目标序列需要 '<START>' 和 '<END>' 标记）
    english_sentences_with_start_end = ["<START> " + seq + " <END>" for seq in english_sentences]
    # print(english_sentences_with_start_end)     # ['<START> hello <END>', '<START> I love programming <END>', '<START> She is happy today <END>', '<START> world peace <END>']
    decoder_inputs, english_seq_len = prepare_encoder_input(english_sentences_with_start_end, english_vocab)

    return encoder_inputs, decoder_inputs, chinese_seq_len, english_seq_len

def train(model, num_epochs=10, lr=0.001):    
    # 词表
    from utils import chinese_vocab
    
    # 损失函数和优化器
    # ignore_index 指定的标签会被忽略，这意味着模型在计算损失和梯度时不会考虑这些标签。这对于处理包含填充（padding）标记的批次非常重要，因为填充标记并不是实际的标签，只是为了使序列对齐。
    criterion = nn.CrossEntropyLoss(ignore_index=chinese_vocab['<PAD>'])
    optimizer = optim.Adam(model.parameters(), lr)    # 优化器将 modal.parameter() 参数作为输入，并在训练过程中更新它们

    # 示例训练循环
    for epoch in range(num_epochs):
        model.train()   # 将模型切换到训练模式

        # 编码器和解码器输入
        encoder_inputs, decoder_inputs, chinese_seq_len, english_seq_len = prepare_data_for_training(training_data, chinese_vocab, english_vocab)
        print("编码器（中文）的输入序列 (数字代表 token 编号): ")
        print(encoder_inputs.size())    # [batch_size, seq_len]
        print("中文序列的 seq_len: ", chinese_seq_len)     
        print("解码器（英文）的输入序列 (数字代表 token 编号): ")
        print(decoder_inputs.size())    # [batch_size, english_seq_len]
        print("英文序列的 seq_len: ", english_seq_len)        
        
        # encoder_inputs : [batch_size, chinese_seq_len]
        # decoder_inputs : [batch_size, english_seq_len]
        target_sequences, tgt_seq_len = prepare_encoder_input(["hello <END>", "I love programming <END>", "She is happy today <END>", "world peace <END>"], english_vocab)
        print("target_sequences.size(): ", target_sequences.size())     # [batch_size, tgt_seq_len]
        print("tgt_seq_len: ", tgt_seq_len)

        # 解码器输入掩码
        # 防止解码器在预测下一个词时看到当前词的未来词。
        tgt_mask = torch.tril(torch.ones(english_seq_len, english_seq_len)).unsqueeze(0).unsqueeze(0)  # [1, 1, english_seq_len, english_seq_len], 运算时会被广播成 [batch_size, n_head, english_seq_len, english_seq_len]
        # 编码器输入掩码
        # src_mask 是为了屏蔽 <pad> 或者填充标记（通常是 0），以防止这些填充位置对注意力计算产生影响。
        # 这种方式仅屏蔽填充标记，允许模型处理 <Unknown>，并学习如何对其进行预测或推理
        src_mask = (encoder_inputs != chinese_vocab["<PAD>"]).unsqueeze(1).unsqueeze(2)   # [batch_size, 1, 1, chinese_seq_len], 运算时会被广播成 [batch_size, n_head, chinese_seq_len, chinese_seq_len]

        # 前向传播
        output = model(encoder_inputs, decoder_inputs, src_mask, tgt_mask)
        print("output.size(): ", output.size())    # [batch_size, english_seq_len, vocab_size]

        # 计算损失
        print("output.view(-1, output.size(-1).size(): ",  output.view(-1, output.size(-1)).size())
        print("target_sequences.view(-1).size(): ", target_sequences.view(-1).size())
        print("因为一定有 english_seq_len > tgt_seq_len, 为了计算交叉熵, 使用 tgt_seq_len 裁剪 output")
        # 裁剪 output 使其与 target_sequences 的长度匹配
        output = output[:, :tgt_seq_len, :]  # 保留前 tgt_seq_len 个 token
        loss = criterion(output.reshape(-1, output.size(-1)), target_sequences.reshape(-1))

        # 反向传播和优化
        optimizer.zero_grad()   # 在计算新的梯度之前，首先将之前的梯度清除。
        loss.backward()         # 计算新梯度
        optimizer.step()        # 使用计算得到的梯度更新模型的参数。

        print(f"Epoch {epoch+1}, Loss: {loss.item()}")


def inference(model, sentence, chinese_vocab, english_vocab, max_len=50):
    model.eval()  # 切换到推理模式，禁用 dropout
    
    # 将输入的中文句子编码为 token 序列
    encoder_input, chinese_seq_len = prepare_encoder_input([sentence], chinese_vocab)  # [1, seq_len]
    
    # 初始化解码器输入：以 <START> 开始
    decoder_input = torch.tensor([[english_vocab['<START>']]], dtype=torch.long)  # [1, 1]
    
    # 编码器输入掩码
    src_mask = (encoder_input != chinese_vocab['<PAD>']).unsqueeze(1).unsqueeze(2)  # [1, 1, 1, seq_len]
    
    output_sentence = []
    
    for i in range(max_len):
        # 生成解码器的 tgt_mask
        tgt_mask = torch.tril(torch.ones(i+1, i+1)).unsqueeze(0).unsqueeze(0)  # [1, 1, i+1, i+1]   # [batch_size, n_head, english_seq_len, english_seq_len]
        
        # 进行前向传播，获取输出
        output = model(encoder_input, decoder_input, src_mask, tgt_mask)  # [1, i+1, vocab_size]    # [batch_size, english_seq_len, vocab_size]
        
        # 取最后一个时间步的输出，并通过 softmax 选择概率最大的 token
        # output[:, -1, :]      # [1, vocab_size]
        # output[:, -1, :].argmax(dim=-1)  # [1], argmax(dim=-1) 的作用是沿着 vocab_size 维度（即词汇表的所有可能词）找到概率最大的那个词的索引。
        # item() 将张量转换为一个 Python 标量值
        next_token = output[:, -1, :].argmax(dim=-1).item()  # [1]
        
        # 将生成的 token 添加到输出句子中
        output_sentence.append(next_token)
        
        # # 如果生成了 <END> 标记，则停止
        # if next_token == english_vocab['<END>']:
        #     break
        
        # 如果生成了 <END> 标记，则停止
        if next_token == english_vocab['<END>']:
            print("我是在自回归过程中打印的语句!")
            break
        
        # 更新 decoder_input，将新生成的 token 加入
        decoder_input = torch.cat([decoder_input, torch.tensor([[next_token]], dtype=torch.long)], dim=-1)  # [1, english_seq_len + 1]
    
    # 反转词汇表：索引 -> 单词
    english_vocab = {idx: word for word, idx in english_vocab.items()}
    # 将生成的 token 序列转换为单词
    translated_sentence:list[str] = [english_vocab.get(token, "<UNKNOWN>") for token in output_sentence]
    
    return ' '.join(translated_sentence)




if __name__ == '__main__':

    # 示例训练数据：中文 -> 英文
    training_data = [
        ("你 好", "hello"),
        ("我 爱 编程", "I love programming"),
        ("她 今天 很 开心", "She is happy today"),
        ("世界 和平", "world peace"),
    ]


    # -------------- 训练模型 --------------

    import torch
    import torch.nn as nn
    import torch.optim as optim

    # 超参数
    # 网络
    vocab_size = 10000  # 词汇表大小
    d_model = 512       # 模型维度
    seq_len = None       # 序列长度, 如果为None, 则计算并使用 batch 中最长的序列长度
    max_len = 5000      # 位置编码所支持的最大序列长度
    batch_size = 3      # 批次大小
    n_head = 8          # 多头注意力的头数
    dropout = 0.1       # dropout 率
    encoder_layer_nums = 6  # 编码器层数量
    decoder_layer_nums = 6  # 解码器层数量
    d_ff = 2024;  # 前馈网络的隐藏层大小
    # 训练
    num_epochs = 10  # 训练轮数
    lr = 0.001      # 学习率
    
    # 模型实例化
    from transformer import Encoder, Decoder, Transformer
    encoder = Encoder(d_model, vocab_size, max_len, n_head, d_ff, encoder_layer_nums, dropout)
    decoder = Decoder(d_model, vocab_size, max_len, n_head, d_ff, decoder_layer_nums, dropout)  
    model = Transformer(encoder, decoder)
    
    # 训练
    train(model, num_epochs, lr)
    
    #推理
    input_sentence = "我 爱 编程"
    output = inference(model, input_sentence, chinese_vocab, english_vocab)
    print("翻译结果: ", output)
    