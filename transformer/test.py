import torch
import torch.nn as nn

# 假设词汇表大小为 1000，嵌入维度为 50
vocab_size = 1000
embedding_dim = 5

# 创建嵌入层
embedding = nn.Embedding(vocab_size, embedding_dim)

# 词汇表映射
vocab = {'I': 0, 'Love': 1, 'You': 2}

# 文本序列
text = "I Love You"

# 将文本转换为索引
indices = [vocab[word] for word in text.split()]
indices_tensor = torch.tensor(indices)  # 转换为张量

# 获取嵌入向量
embeddings = embedding(indices_tensor)

# 查看嵌入矩阵
print("嵌入矩阵: ")
print(embedding.weight.size())  


# 查看结果
print("Indices: ", indices)
print("Embeddings: ", embeddings)
