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
    # 分别处理中文和英文句子
    chinese_sentences, english_sentences = zip(*training_data)
    # print(*training_data)   # ('你好', 'hello') ('我爱编程', 'I love programming') ('她今天很开心', 'She is happy today') ('世界和平', 'world peace')
    # print(chinese_sentences)   # ('你好', '我爱编程', '她今天很开心', '世界和平')


    # 准备编码器（中文）的输入
    encoder_inputs, seq_len = prepare_encoder_input(chinese_sentences, chinese_vocab)
    print("编码器（中文）的输入序列 (数字代表 token 编号): ")
    print(encoder_inputs)
    print("中文序列的 seq_len: ", seq_len)        

    # 准备解码器（英文）的输入（目标序列需要 '<START>' 和 '<END>' 标记）
    english_sentences_with_start_end = ["<START> " + seq + " <END>" for seq in english_sentences]
    # print(english_sentences_with_start_end)     # ['<START> hello <END>', '<START> I love programming <END>', '<START> She is happy today <END>', '<START> world peace <END>']
    decoder_inputs, _ = prepare_encoder_input(english_sentences_with_start_end, english_vocab)
    print("解码器（英文）的输入序列 (数字代表 token 编号): ")
    print(decoder_inputs)
    print("英文序列的 seq_len: ", _)        

    return encoder_inputs, decoder_inputs


if __name__ == '__main__':

    # 示例训练数据：中文 -> 英文
    training_data = [
        ("你好", "hello"),
        ("我 爱 编程", "I love programming"),
        ("她 今天 很 开心", "She is happy today"),
        ("世界 和平", "world peace"),
    ]

    # 编码
    encoder_inputs, decoder_inputs = prepare_data_for_training(training_data, chinese_vocab, english_vocab)
    
    # 假设 target_sequences 是解码器输出中的下一个词
    target_sequences, _ = prepare_encoder_input(["hello <END>", "I love programming <END>", "She is happy today <END>", "world peace <END>"], english_vocab)
    print("target sequences: ")
    print(target_sequences)

    # -------------- 训练模型 --------------

    import torch
    import torch.nn as nn
    import torch.optim as optim

    # 假设已经定义了 Encoder, Decoder, 和 Transformer
    encoder = Encoder(...)
    decoder = Decoder(...)
    model = Transformer(encoder, decoder)

    # 损失函数和优化器
    # ignore_index 指定的标签会被忽略，这意味着模型在计算损失和梯度时不会考虑这些标签。这对于处理包含填充（padding）标记的批次非常重要，因为填充标记并不是实际的标签，只是为了使序列对齐。
    criterion = nn.CrossEntropyLoss(ignore_index=chinese_vocab['<PAD>'])
    optimizer = optim.Adam(model.parameters(), lr=0.001)    # 优化器将 modal.parameter() 参数作为输入，并在训练过程中更新它们

    # 示例训练循环
    for epoch in range(num_epochs):
        model.train()   # 将模型切换到训练模式

        # 编码器和解码器输入
        encoder_inputs, decoder_inputs = prepare_data_for_training(training_data, chinese_vocab, english_vocab)
        target_sequences, _ = prepare_encoder_input(["hello <END>", "I love programming <END>", "She is happy today <END>", "world peace <END>"], english_vocab)

        # 前向传播
        encoder_output = encoder(encoder_inputs)
        decoder_output = decoder(decoder_inputs, encoder_output)

        # 计算损失
        loss = criterion(decoder_output.view(-1, decoder_output.size(-1)), target_sequences.view(-1))

        # 反向传播和优化
        optimizer.zero_grad()   # 在计算新的梯度之前，首先将之前的梯度清除。
        loss.backward()         # 计算新梯度
        optimizer.step()        # 使用计算得到的梯度更新模型的参数。

        print(f"Epoch {epoch+1}, Loss: {loss.item()}")
