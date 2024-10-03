import numpy as np
from sklearn.preprocessing import normalize

# 定义超参数
n = 5  # batch size
d_i = 512  # image embedding size
d_t = 1024  # text embedding size
d_e = 256  # joint embedding size
t = 0.07  # learned temperature parameter

# 随机初始化权重矩阵
W_i = np.random.randn(d_i, d_e)  # image projection matrix
W_t = np.random.randn(d_t, d_e)  # text projection matrix

# 模拟 image_encoder 和 text_encoder 的输出 (在实际中使用预训练模型)
def image_encoder(I):
    return np.random.randn(I.shape[0], d_i)   # [n, d_i]

def text_encoder(T):
    return np.random.randn(T.shape[0], d_t)   # [n, d_t]

# 随机生成模拟的图像和文本数据
I = np.random.randn(n, 224, 224, 3)  # image batch [n, h, w, c]
T = np.random.randn(n, 77)  # text batch [n, l]

# 提取每种模态的特征表示
I_f = image_encoder(I)  # [n, d_i]
T_f = text_encoder(T)  # [n, d_t]

# 计算 joint multimodal embedding [n, d_e]
I_e = normalize(np.dot(I_f, W_i), axis=1)  # L2 normalization
T_e = normalize(np.dot(T_f, W_t), axis=1)  # L2 normalization

# 计算缩放后的 pairwise cosine similarities [n, n]
logits = np.dot(I_e, T_e.T) * np.exp(t)

# 对称的损失函数
labels = np.arange(n)
def cross_entropy_loss(logits, labels, axis):
    '''
    解读：
    ------
    通过 probs[np.arange(n), labels]，我们获取 probs 中每一行中对应标签的概率。例如，假设 probs 是一个 3x3 的矩阵：
    probs = np.array([[0.2, 0.5, 0.3],
                  [0.1, 0.6, 0.3],
                  [0.3, 0.4, 0.3]])

    labels = np.array([1, 0, 2])  # 正确的类别索引
    probs[np.arange(3), labels] 将得到 probs 中每行索引对应的元素：
    probs[np.arange(3), labels] = [0.5, 0.1, 0.3]
    即，对于第一个样本，选择第 1 列的概率 0.5；对于第二个样本，选择第 0 列的概率 0.1；对于第三个样本，选择第 2 列的概率 0.3。
    np.arange(n) 这个参数的作用是为每一行的每个元素生成下标。
    labels 参数是规定每一行取哪一个下标的元素的规则
    '''
    exp_logits = np.exp(logits)     # size: (n, n)
    probs = exp_logits / np.sum(exp_logits, axis=axis, keepdims=True)   # (n, n) / (n, n) = (n, n)
    log_probs = -np.log(probs[np.arange(n), labels])
    # print("log_probs.size(): ", log_probs.shape)  # size: (n,)  (标量)
    return np.mean(log_probs)

# 计算图像和文本的损失
loss_i = cross_entropy_loss(logits, labels, axis=0)     # size: () 标量
loss_t = cross_entropy_loss(logits, labels, axis=1)     # size: () 标量

# 最终的损失为两者的平均
loss = (loss_i + loss_t) / 2    # size: 标量

print("Final loss:", loss)
