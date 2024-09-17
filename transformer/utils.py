import torch

# 定义词汇表和转换函数
vocab = {
    '<PAD>': 0,
    '<START>': 1,
    '<END>': 2,
    '<Unknown>': 3,
    'hello': 4,
    'world': 5,
    'hi': 6,
    'I': 7,
    'love': 8,
    'U': 9,
    'and': 10,
    'me': 11,
    'today': 12,
    'She': 13,
}

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

def text_to_indices(text, vocab):
    """
    将文本转换为词汇表中的索引序列，未知的词汇用 '<Unknown>' 表示
    """
    return [vocab.get(word, vocab["<Unknown>"]) for word in text.split()]

def pad_sequences(sequences, pad_value, max_len=None):
    '''
    - sequences: 二维列表，每个元素是一个序列, [batch_size, seq_len]
    - pad_value: 填充值
    - max_len: 填充后的序列的最大长度, 如果为 None, 则使用最长的序列长度
    '''
    if max_len is None:
        max_len = max(len(seq) for seq in sequences)
    padded_sequences = [seq + [pad_value] * (max_len - len(seq)) for seq in sequences]
    return padded_sequences, max_len

def prepare_encoder_input(sequences:list[str], vocab:dict, seq_len:int=None):
    """
    将文本序列转换为编码器的输入张量，并将其标准化为相同长度。
    
    参数：
    - sequences: 文本序列的列表，每个元素是一个字符串，代表一个句子。
    - vocab: 词汇表，字典类型，键为词汇，值为该词汇在词汇表中的索引编号。
    - seq_len: 序列的长度，如果为 None，函数将使用输入序列中的最长序列长度作为标准长度进行填充。
    
    返回：
    - tensor_sequences: 转换后的序列张量，形状为 [batch_size, seq_len]，其中每个序列的长度为 seq_len。
    - seq_len: 序列长度，如果未提供，则返回计算出的最长序列长度。
    """
    # 分词
    print("分词:")
    print([seq.split() for seq in sequences])
    
    # 转换为索引
    indexed_sequences = [text_to_indices(seq, vocab) for seq in sequences]
    
    # 填充序列
    padded_sequences, seq_len = pad_sequences(indexed_sequences, pad_value=vocab["<PAD>"], max_len=seq_len)
    
    # 将填充后的序列转换为张量
    tensor_sequences = torch.tensor(padded_sequences, dtype=torch.long)
    
    return tensor_sequences, seq_len


def predict_words(softmax_output, inverse_vocab, seq_len) -> list[str]:
    """
    将解码器的输出转换为词汇表中的预测词汇。
    
    参数：
    - softmax_output: softmax层的输出张量，形状为  [batch_size, seq_len, vocab_size]
    - inverse_vocab: 反转词汇表，字典类型，索引到词汇的映射 e.g.{0: word, 1 : word}
    - seq_len: 序列长度，用于将预测的索引恢复成序列
    
    返回：
    - predicted_words: 预测的词汇列表，按句子划分 [batch_size(), seq_len]
    """
    
    # 获取预测词汇的索引
    _, predicted_indices = torch.max(softmax_output, dim=-1)  # predicted_indices : [batch_size, seq_len]
    # print("predicted_indices.size(): ", predicted_indices.size())
    # print("predicted_indices: ")
    # print(predicted_indices)
    
    
    # 将预测的索引转换为实际的词汇
    predicted_words = [inverse_vocab.get(idx.item(), '<Unknown>') for idx in predicted_indices.view(-1)]  # view(-1) 表示转成行向量, idx.item() 将单个元素的张量（tensor）转换为 Python 的标量
    
    # print("predicted_words: ")
    # print(predicted_words)
    # print("seq_len: ", seq_len)
        
    predicted_words = [predicted_words[i:i + seq_len] for i in range(0, len(predicted_words), seq_len)]
    
    return predicted_words


import torch
import torch.nn.functional as F

class CrossEntropyLossCustom:
    def __init__(self):
        pass

    def __call__(self, logits, targets):
        """
        计算交叉熵损失
        
        参数:
        - logits: 模型输出的预测值，形状为 [batch_size, seq_len, vocab_size]
        - targets: 真实标签，形状为 [batch_size, seq_len]
        
        返回:
        - 损失值，标量
        """
        # 将 logits 展平为 [batch_size * seq_len, vocab_size]
        batch_size, seq_len, vocab_size = logits.size()
        logits_flat = logits.view(-1, vocab_size)
        
        # 将 targets 展平为 [batch_size * seq_len]
        targets_flat = targets.view(-1)
        
        # 计算交叉熵损失 - 简单实现版
        loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=-1)       
        
        # 计算交叉熵损失 - 手撕版
        # softmax, 将 logits 转换为概率分布
        # 再 log(softmax)
        # log_probs = F.log_softmax(logits_flat, dim=-1)  # [batch_size * seq_len, vocab_size]
        # # 
        # index = targets_flat.unsqueeze(1)  # [batch_size * seq_len, 1]
        # # 选择正确标签的对数概率
        # # 使用 gather 函数选择 logits 中对应目标的对数概率
        # # gather 输入形状为 [batch_size * seq_len, vocab_size]
        # # index 形状为 [batch_size * seq_len]
        # # torch.gather 根据 index 沿着 dim 从 log_probs 中提取每个样本的正确类别的对数概率。
        # selected_log_probs = torch.gather(log_probs, dim=-1, index=index)  # [batch_size * seq_len, 1]
        # # print("selected_log_probs: ")
        # # print(selected_log_probs)
        # loss = (-1) * selected_log_probs.mean();        
        
        return loss
    


if __name__ == "__main__":
    # # --------- 编码测试 --------------
    # sequences = ["hello world", "hi I love U", "She and me today will"]
    # indices, seq_len = prepare_encoder_input(sequences, vocab, seq_len=None)
    # print("indices: ")
    # print(indices)
    # print("seq_len: ", seq_len)
    
    # -------- predict_words 测试 ---------
    # softmax_output = torch.tensor([
    #                                 [[0.2, 0.3, 0.4, 0.7, 0.6],[0.2, 0.3, 0.4, 0.5, 0.6], [0.7, 0.3, 0.4, 0.5, 0.6]],
    #                                 [[0.2, 0.7, 0.4, 0.5, 0.6], [0.2, 0.3, 0.4, 0.7, 0.6], [0.7, 0.3, 0.4, 0.5, 0.6]]
    #                               ])
    # print("softmax_output.size(): ", softmax_output.size())     # [batch_size, seq_len, vocab_size]
    # #  seq1: 3, 4, 0 -> <Unknown> hello <PAD>
    # #  seq2: 1, 3, 0 -> <START> <Unknown> <PAD>
    
    # #  定义反向词汇表 (for word prediction)
    # inverse_vocab = {index: word for word, index in vocab.items()}
    # #
    # predicted_words = predict_words(softmax_output, inverse_vocab, seq_len=3)
    # print(predicted_words)
    
    # for i, sentence in enumerate(predicted_words):
    #     print(f"Sentence {i+1}: {' '.join(sentence)}")
        
        
    # --------  交叉熵测试 ---------
    # 创建一个自定义损失函数实例
    criterion = CrossEntropyLossCustom()

    # 假设有一个 batch_size = 2, seq_len = 3, vocab_size = 5
    logits = torch.tensor([
        [[0.2, 0.3, 0.4, 0.5, 0.6], [0.2, 0.3, 0.4, 0.5, 0.6], [0.1, 0.2, 0.3, 0.4, 0.5]],
        [[0.5, 0.4, 0.3, 0.2, 0.1], [0.6, 0.5, 0.4, 0.3, 0.2], [0.1, 0.2, 0.3, 0.4, 0.5]]
    ], dtype=torch.float)

    # label : [batch_size, seq_len]
    targets = torch.tensor([
        [4, 0, 1],
        [2, 3, 4]
    ], dtype=torch.long)

    # 计算损失
    loss = criterion(logits, targets)
    print("计算的损失值: ", loss.item())
    
    
    