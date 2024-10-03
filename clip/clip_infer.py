'''
假设输入为一张图片和三个文本描述。
推理过程中，我们将使用预训练好的 image_encoder 提取图像的特征向量，
使用 text_encoder 提取文本的特征向量，然后计算图像与每个文本之间的相似度（如余弦相似度），
并返回最相似的文本。
'''


import numpy as np
from sklearn.preprocessing import normalize

# 定义超参数
d_i = 512  # image embedding size
d_t = 512  # text embedding size
d_e = 256  # joint embedding size
t = 0.07   # learned temperature parameter

# 随机初始化权重矩阵 (实际推理中应为训练好的模型参数)
W_i = np.random.randn(d_i, d_e)  # image projection matrix
W_t = np.random.randn(d_t, d_e)  # text projection matrix

# 模拟 image_encoder 和 text_encoder 的输出 (实际中应使用预训练模型)
def image_encoder(I):
    return np.random.randn(1, d_i)  # 单张图片返回特征向量 [1, d_i]

def text_encoder(T):
    return np.random.randn(len(T), d_t)  # 返回多条文本的特征 [n, d_t]

# 模拟推理的图像和文本
image = np.random.randn(224, 224, 3)  # 单张图片
texts = ["A cat sitting on the grass", "A dog playing with a ball", "A bird flying in the sky"]  # 3个文本描述

# 提取图像的特征表示
I_f = image_encoder(image)  # [1, d_i]

# 提取文本的特征表示
T_f = text_encoder(texts)  # [3, d_t]

# 计算 joint multimodal embedding
I_e = normalize(np.dot(I_f, W_i), axis=1)  # 图像嵌入 [1, d_e]
T_e = normalize(np.dot(T_f, W_t), axis=1)  # 文本嵌入 [3, d_e]

# 计算图像与每个文本的相似度（余弦相似度）
logits = np.dot(I_e, T_e.T) * np.exp(t)  # [1, 3] 每个文本的相似度

# 输出最相似的文本
most_similar_index = np.argmax(logits)  # 找到相似度最高的文本索引
most_similar_text = texts[most_similar_index]

# 输出相似度结果
print("Text descriptions:", texts)
print("Similarity scores:", logits.flatten())
print("Most similar text:", most_similar_text)
