import torch
import torch.nn as nn
import torch.nn.functional as F

class MoCo(nn.Module):
    def __init__(self, encoder, dim=128, K=4096, m=0.999, tau=0.07):
        """
        encoder: 任意 backbone (如 ResNet, Transformer)
        dim: 投影维度
        K: 队列容量 (负样本数量)
        m: 动量更新系数
        tau: 温度
        """
        super().__init__()
        self.K = K
        self.m = m
        self.tau = tau

        # 创建 query encoder 和 key encoder（参数复制）
        self.encoder_q = encoder
        self.encoder_k = encoder.__class__()         # 创建同结构新模型
        self.encoder_k.load_state_dict(encoder.state_dict())

        # key encoder 不训练
        for p in self.encoder_k.parameters():
            p.requires_grad = False

        # 队列：存放负样本 keys    (dim, K)
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = F.normalize(self.queue, dim=0)  # 列归一化
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        key_encoder = m * key_encoder + (1-m) * query_encoder
        """
        for q_param, k_param in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            k_param.data = k_param.data * self.m + q_param.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """
        将新的 keys 入队，旧 keys 出队，维护队列为 FIFO
        keys: (N, dim)
        """
        N = keys.shape[0]
        ptr = int(self.queue_ptr)

        # 一次性覆盖前 N 个位置
        self.queue[:, ptr:ptr + N] = keys.T
        ptr = (ptr + N) % self.K   # 循环队列
        self.queue_ptr[0] = ptr

    def forward(self, im_q, im_k):
        """
        输入两种增强图像 im_q 和 im_k
        输出 loss
        """
        # 1) 计算 q
        q = self.encoder_q(im_q)           # (N, dim)
        q = F.normalize(q, dim=1)

        # 2) 更新动量 key encoder
        with torch.no_grad():
            self._momentum_update_key_encoder()

            k = self.encoder_k(im_k)       # (N, dim)
            k = F.normalize(k, dim=1)

        # 3) MoCo InfoNCE loss
        # 正样本：bmm(q·k+ )
        pos = torch.bmm(q.unsqueeze(1), k.unsqueeze(2))  # (N,1,1)
        pos = pos.squeeze(2)  # (N,1)
        '''
        等价1：
        pos = torch.einsum('nd,nd->n', q, k).unsqueeze(1)
        等价2：
        # 正样本：q·k+   (逐行点积)
        pos = torch.bmm(q.unsqueeze(1), k.unsqueeze(2))  # (N,1,1)
        '''

        # 负样本：q·queue
        neg = torch.einsum("nd,dk->nk", [q, self.queue])  # (N,K)

        # 拼 logits: [pos | neg]
        logits = torch.cat([pos, neg], dim=1) / self.tau  # (N, 1+K)

        labels = torch.zeros(logits.size(0), dtype=torch.long).to(logits.device)  # (N,)
        loss = F.cross_entropy(logits, labels)

        # 4) 更新队列
        self._dequeue_and_enqueue(k)

        return loss
