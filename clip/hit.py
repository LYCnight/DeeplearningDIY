import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ------------------------------
# 1. Transformer Encoder (shared)
# ------------------------------
class TinyTransformer(nn.Module):
    def __init__(self, dim=256, depth=4, heads=4):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=dim, nhead=heads, batch_first=True
            ) for _ in range(depth)
        ])
        self.dim = dim

    def forward(self, x):
        outputs = []
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            outputs.append(x)
        return outputs  # return all layers


# ------------------------------
# 2. Video Encoder
# ------------------------------
class VideoEncoder(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.encoder = TinyTransformer(dim=dim, depth=4)

        self.proj_feature = nn.Linear(dim, dim)
        self.proj_semantic = nn.Linear(dim, dim)

    def forward(self, x):
        """
        x: (B, T, dim)
        """
        layers = self.encoder(x)

        # low-layer: feature-level
        feat = layers[0].mean(dim=1)
        feat = self.proj_feature(feat)

        # last-layer: semantic-level
        sem = layers[-1].mean(dim=1)
        sem = self.proj_semantic(sem)

        return F.normalize(feat, dim=-1), F.normalize(sem, dim=-1)


# ------------------------------
# 3. Text Encoder
# ------------------------------
class TextEncoder(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.encoder = TinyTransformer(dim=dim, depth=4)

        self.word_proj = nn.Linear(dim, dim)
        self.sem_proj = nn.Linear(dim, dim)

    def forward(self, x):
        layers = self.encoder(x)

        # word-level: layer 1
        word = layers[0].mean(dim=1)
        word = self.word_proj(word)

        # semantic-level: last layer
        sem = layers[-1].mean(dim=1)
        sem = self.sem_proj(sem)

        return F.normalize(word, dim=-1), F.normalize(sem, dim=-1)


# ------------------------------
# 4. Momentum Encoder (MoCo)
# ------------------------------
def momentum_update(query_net, key_net, m=0.999):
    for param_q, param_k in zip(query_net.parameters(), key_net.parameters()):
        param_k.data = param_k.data * m + param_q.data * (1 - m)


# ------------------------------
# 5. Memory Bank
# ------------------------------
class MemoryBank:
    def __init__(self, size=4096, dim=256):
        self.size = size
        self.bank = torch.randn(size, dim)
        self.ptr = 0

    @torch.no_grad()
    def enqueue(self, feats):
        b = feats.size(0)
        if b > self.size:
            feats = feats[-self.size:]

        end = self.ptr + b
        if end <= self.size:
            self.bank[self.ptr:end] = feats
        else:
            first = self.size - self.ptr
            self.bank[self.ptr:] = feats[:first]
            self.bank[:end - self.size] = feats[first:]
        self.ptr = (self.ptr + b) % self.size

    def get_all(self):
        return self.bank.clone()


# ------------------------------
# 6. InfoNCE loss
# ------------------------------
def contrastive_loss(q, k_pos, neg_bank, temp=0.07):
    # q: (B, dim)
    # k_pos: (B, dim)
    # neg_bank: (K, dim)
    pos = torch.sum(q * k_pos, dim=-1, keepdim=True)  # (B,1)
    neg = q @ neg_bank.t()  # (B, K)

    logits = torch.cat([pos, neg], dim=1) / temp
    labels = torch.zeros(q.size(0), dtype=torch.long, device=q.device)

    return F.cross_entropy(logits, labels)


# ------------------------------
# 7. HiT Model (minimal)
# ------------------------------
class HiT(nn.Module):
    def __init__(self, dim=256, bank_size=4096):
        super().__init__()
        # Query encoders
        self.video_q = VideoEncoder(dim)
        self.text_q = TextEncoder(dim)

        # Key encoders
        self.video_k = VideoEncoder(dim)
        self.text_k = TextEncoder(dim)

        # initialize key encoders
        for p_q, p_k in zip(self.video_q.parameters(), self.video_k.parameters()):
            p_k.data.copy_(p_q.data)
        for p_q, p_k in zip(self.text_q.parameters(), self.text_k.parameters()):
            p_k.data.copy_(p_q.data)

        # Memory banks
        self.bank_t_word = MemoryBank(size=bank_size, dim=dim)
        self.bank_t_sem = MemoryBank(size=bank_size, dim=dim)
        self.bank_v_feat = MemoryBank(size=bank_size, dim=dim)
        self.bank_v_sem = MemoryBank(size=bank_size, dim=dim)

    def forward(self, v, t):
        # Query encoders
        v_f_q, v_s_q = self.video_q(v)
        t_w_q, t_s_q = self.text_q(t)

        # Key encoders
        with torch.no_grad():
            v_f_k, v_s_k = self.video_k(v)
            t_w_k, t_s_k = self.text_k(t)

        return (v_f_q, v_s_q, t_w_q, t_s_q,
                v_f_k, v_s_k, t_w_k, t_s_k)

    def update_momentum(self):
        momentum_update(self.video_q, self.video_k)
        momentum_update(self.text_q, self.text_k)


# ------------------------------
# 8. Dummy Training Loop
# ------------------------------
def train():
    dim = 256
    model = HiT(dim=dim, bank_size=2048)
    model.train()

    optim = torch.optim.Adam(model.parameters(), lr=1e-4)

    for step in range(200):
        # Dummy batch: video token seq & text token seq
        B, T = 32, 20
        video = torch.randn(B, T, dim)
        text = torch.randn(B, T, dim)

        (v_f_q, v_s_q, t_w_q, t_s_q,
         v_f_k, v_s_k, t_w_k, t_s_k) = model(video, text)

        # Build losses
        loss_feat_v2t = contrastive_loss(v_f_q, t_w_k, model.bank_t_word.get_all())
        loss_feat_t2v = contrastive_loss(t_w_q, v_f_k, model.bank_v_feat.get_all())
        loss_sem_v2t = contrastive_loss(v_s_q, t_s_k, model.bank_t_sem.get_all())
        loss_sem_t2v = contrastive_loss(t_s_q, v_s_k, model.bank_v_sem.get_all())

        loss = loss_feat_v2t + loss_feat_t2v + loss_sem_v2t + loss_sem_t2v

        optim.zero_grad()
        loss.backward()
        optim.step()

        # Momentum update & enqueue
        model.update_momentum()
        with torch.no_grad():
            model.bank_t_word.enqueue(t_w_k)
            model.bank_t_sem.enqueue(t_s_k)
            model.bank_v_feat.enqueue(v_f_k)
            model.bank_v_sem.enqueue(v_s_k)

        if step % 20 == 0:
            print(f"[step {step}] loss={loss.item():.4f}")

    print("Training done.")


if __name__ == "__main__":
    train()



