import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Optional, Dict


# DHCP
class DynamicPrototypeBank(nn.Module):
    """
    简化版 DHCP：
    - 维护一个可学习 prototype 矩阵 A: [M, D]
    - 支持根据输入特征 e_i 做 attention-based momentum update
    - 支持从 query 中计算到各个 prototype 的 softmax 权重
    """
    def __init__(self, num_prototypes: int, dim: int, momentum: float = 0.99, init_scale: float = 0.02):
        super().__init__()
        self.num_prototypes = num_prototypes
        self.dim = dim   # 32
        self.momentum = momentum

        # A: [M, D], 这里直接随机初始化，实际可从 CLIP + 聚类初始化
        self.prototypes = nn.Parameter(init_scale * torch.randn(num_prototypes, dim))  # (M, D)
        self.register_buffer("temperature", torch.tensor(0.07))  # τ

    @torch.no_grad()
    def momentum_update(self, feats: Tensor):
        """
        feats: [N, D]   e_i in paper
        用 attention + momentum 更新 prototype
        """
        if feats.numel() == 0:
            return

        feats = F.normalize(feats, dim=-1)                  # [N, D]
        prot = F.normalize(self.prototypes.data, dim=-1)    # [M, D]

        # cos sim: [N, M]
        logits = feats @ prot.t() / self.temperature        # [N, M]   # (5,10)
        weights = F.softmax(logits, dim=-1)                 # D_{i,j}   # (N, M)   # (5, 10)

        # ====== 添加断点 ======
        import ipdb; ipdb.set_trace()

        # 汇总到 prototype: [M, D]
        agg = weights.t() @ feats                           # sum_i D_{i,j} e_i
        # (M, N) @ (N, D) = (M, D)   # (10, 32)

        # momentum update
        self.prototypes.data = (
            self.momentum * self.prototypes.data
            + (1 - self.momentum) * F.layer_norm(agg, (self.dim,))
        )

    def forward(self, queries: Tensor) -> Tensor:
        """
        queries: [B, N, D]  (已经在 CLIP 空间)
        return: weights: [B, N, M]  每个 query 对 prototype 的 softmax 权重 w_{n,j}
        """
        B, N, D = queries.shape
        q = F.normalize(queries, dim=-1)                    # [B, N, D]
        p = F.normalize(self.prototypes, dim=-1)            # [M, D]

        # [B, N, M]
        logits = torch.einsum("bnd,md->bnm", q, p) / self.temperature
        weights = F.softmax(logits, dim=-1)
        return weights


class ProjectionHead(nn.Module):
    """
    h_θ: 将 DETR decoder 的 query 维度 C -> CLIP 维度 D
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, N, C]
        return self.proj(x)  # [B, N, D]


# Hi-Know DPA --------------------

class SemanticEnhancer(nn.Module):
    """
    Hi-Know DPA 核心：
    - 输入投影后的 q_hat: [B, N, D]
    - prototypeBank 提供 prototype A: [M, D]（隐含在 bank.prototypes 中）
    - 计算 w_{n,j}，再聚合得到 r_n
    - 可选：用 teacher（CLIP）提供 target w̃_n 做 KL 蒸馏
    """
    def __init__(self, dim: int, prototype_bank: DynamicPrototypeBank):
        super().__init__()
        self.dim = dim
        self.prototype_bank = prototype_bank
        self.alpha = nn.Parameter(torch.tensor(0.07))  # 温度 α
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def _get_student_weights(self, q_hat: Tensor) -> Tensor:
        """
        q_hat: [B, N, D]
        return: w_{n,j}: [B, N, M]
        """
        B, N, D = q_hat.shape
        q = F.normalize(q_hat, dim=-1)
        p = F.normalize(self.prototype_bank.prototypes, dim=-1)  # [M, D]

        logits = torch.einsum("bnd,md->bnm", q, p) / self.alpha
        w = F.softmax(logits, dim=-1)
        return w

    def _get_teacher_weights(self, q_hat: Tensor, text_prototypes: Tensor) -> Tensor:
        """
        用 Teacher (CLIP text embeddings) 产生 target 分布 w̃_n
        q_hat: [B, N, D] (仍在 CLIP 空间)
        text_prototypes: [M_text, D] (类别名 + LLaVA phrases)
        """
        B, N, D = q_hat.shape
        q = F.normalize(q_hat, dim=-1)
        t = F.normalize(text_prototypes, dim=-1)

        logits = torch.einsum("bnd,md->bnm", q, t) / self.alpha
        w_t = F.softmax(logits, dim=-1)  # [B, N, M_text]
        return w_t

    def forward(
        self,
        q_hat: Tensor,
        text_prototypes: Optional[Tensor] = None,
        lambda_kl: float = 1.0,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        q_hat: [B, N, D]    h_θ(Q)
        text_prototypes: [M_text, D] 用于 teacher 分布（可选）
        return:
            r: [B, N, D]  语义增强后的 query
            loss_dict: {"loss_kl": ...}
        """
        # 学生对 DHCP prototype 的注意力
        w_student = self._get_student_weights(q_hat)        # [B, N, M]
        A = self.prototype_bank.prototypes                  # [M, D]

        # 聚合得到 r_n: Σ w_{n,j} A_j
        r = torch.einsum("bnm,md->bnd", w_student, A)       # [B, N, D]

        # residual + MLP
        r = r + self.mlp(q_hat)                             # [B, N, D]

        loss_kl = q_hat.new_tensor(0.0)
        if text_prototypes is not None:
            # teacher 分布 (w̃_n)
            w_teacher = self._get_teacher_weights(q_hat, text_prototypes)  # [B, N, M_text]

            # 为了简化，这里假设 text_prototypes 的数量与 DHCP prototype 数相等并对齐
            # 如果不相等，你可以先对 text_prototypes 做聚类或线性投影到 M 维。
            if w_teacher.shape[-1] == w_student.shape[-1]:
                # KL( w_student || w_teacher )
                # 按论文： sum_n KL( w_n || w̃_n )
                w_s = w_student.clamp_min(1e-8)
                w_t = w_teacher.clamp_min(1e-8)
                loss_kl = F.kl_div(
                    w_s.log(), w_t, reduction="batchmean"
                ) * lambda_kl

        return r, {"loss_kl": loss_kl}



# PD-DuGi -----------
class PDTHead(nn.Module):
    """
    Parametric Decoupling Transformer (极简版)：
    - 输入 r_n: [B, N, D]
    - 输出 t̃_n: [B, N, C]  (open-vocab logits)
    注意：理论上这里有 cross-attn 到 prototypes，这里用 MLP 版简化。
    """
    def __init__(self, dim: int, num_classes: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.GELU(),
        )
        self.classifier = nn.Linear(dim, num_classes)

    def forward(self, r: Tensor) -> Tensor:
        """
        r: [B, N, D]
        return: logits: [B, N, C]
        """
        x = self.layers(r)
        logits = self.classifier(x)
        return logits


# -------- DeCo-DETR 框架骨架 -----------
class DummyBackbone(nn.Module):
    """占位：返回 encoder feature 和 pos embedding。"""
    def __init__(self, out_dim=256):
        super().__init__()
        self.out_dim = out_dim
        self.conv = nn.Conv2d(3, out_dim, 3, padding=1)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        # x: [B, 3, H, W]
        feat = self.conv(x)     # [B, C, H, W]
        pos = torch.zeros_like(feat)  # 简化掉位置编码
        return feat, pos


class DummyDETRDecoder(nn.Module):
    """占位：简单用一个 TransformerEncoder 模拟 query 更新。"""
    def __init__(self, num_queries=100, d_model=256, nhead=8, num_layers=3):
        super().__init__()
        self.query_embed = nn.Embedding(num_queries, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, feat: Tensor) -> Tensor:
        """
        feat: [B, C, H, W]
        return: queries: [B, N, C]
        """
        B, C, H, W = feat.shape
        N = self.query_embed.num_embeddings

        # flatten 特征
        src = feat.flatten(2).permute(2, 0, 1)      # [HW, B, C]
        # 用 learnable queries 作为 input（非常简化）
        queries = self.query_embed.weight.unsqueeze(1).expand(-1, B, -1)  # [N, B, C]

        out = self.transformer(queries, src)        # [N, B, C]
        out = out.permute(1, 0, 2)                  # [B, N, C]
        return out



# ---------------- DeCo-Detr 主类 ------------------
class DeCoDETR(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_queries: int = 100,
        det_dim: int = 256,
        clip_dim: int = 512,
        num_prototypes: int = 6003,  # 1203 + 4800
    ):
        super().__init__()
        # 1) DETR 主体
        self.backbone = DummyBackbone(out_dim=det_dim)
        self.decoder = DummyDETRDecoder(num_queries=num_queries, d_model=det_dim)

        # 2) DHCP prototype bank
        self.prototype_bank = DynamicPrototypeBank(num_prototypes, clip_dim, momentum=0.99)

        # 3) 投影头 + 语义增强
        self.proj_head = ProjectionHead(in_dim=det_dim, out_dim=clip_dim)
        self.semantic_enhancer = SemanticEnhancer(dim=clip_dim, prototype_bank=self.prototype_bank)

        # 4) PDT + 分类头（open-vocab logits）
        self.pdt_head = PDTHead(dim=clip_dim, num_classes=num_classes)

        # 5) 基础 bbox head（简化，只演示结构）
        self.bbox_head = nn.Linear(det_dim, 4)  # [cx, cy, w, h]

    def forward(self, images: Tensor) -> Dict[str, Tensor]:
        """
        images: [B, 3, H, W]
        return: dict with:
            - logits_det: [B, N, C]  (检测 logits, 简化)
            - boxes: [B, N, 4]
            - logits_pdt: [B, N, C]  (open-vocab logits)
        """
        B = images.size(0)

        # 1) Backbone + decoder
        feat, pos = self.backbone(images)        # feat: [B, C, H, W]
        queries = self.decoder(feat)             # [B, N, C]

        # 2) 标准 DETR head（这里只是示例）
        logits_det = nn.Linear(queries.size(-1), self.pdt_head.classifier.out_features).to(queries.device)
        cls_det = logits_det(queries)            # [B, N, C]
        boxes = self.bbox_head(queries)          # [B, N, 4]

        # 3) 投影到 CLIP 空间
        q_hat = self.proj_head(queries)          # [B, N, D]

        # 4) DHCP 权重 + 语义增强（Hi-Know DPA）
        #   （此处先不传 text_prototypes, 只看结构）
        r, loss_dict = self.semantic_enhancer(q_hat, text_prototypes=None, lambda_kl=1.0)

        # 5) PDT 生成 open-vocab logits
        logits_pdt = self.pdt_head(r)            # [B, N, C]

        return {
            "cls_det": cls_det,
            "boxes": boxes,
            "cls_pdt": logits_pdt,
            **loss_dict,
        }
    


#  ---------------- 训练模块 ----------------
def training_step(
    model: DeCoDETR,
    images: Tensor,
    targets: Dict[str, Tensor],
    text_prototypes: Optional[Tensor],
    lambda_align: float = 1.0,
    optimizer_det=None,
    optimizer_sem=None,
):
    """
    targets 示例：
        {
            "labels": [B, N],  # 对应 DETR 匹配后的类别 id
            "boxes":  [B, N, 4]
        }
    text_prototypes: [M_text, D] CLIP text embeddings (类别名 + LLaVA phrase)
    """

    out = model(images)
    cls_det = out["cls_det"]      # [B, N, C]
    pred_boxes = out["boxes"]     # [B, N, 4]
    cls_pdt = out["cls_pdt"]      # [B, N, C]
    loss_kl = out["loss_kl"]      # 标量 (若未使用 teacher 则为 0)

    labels = targets["labels"]    # [B, N]
    gt_boxes = targets["boxes"]   # [B, N, 4]

    # -------------------------
    # 1) Detection loss (L_det)
    # -------------------------
    # 分类：CE
    det_loss_cls = F.cross_entropy(
        cls_det.flatten(0, 1),    # [B*N, C]
        labels.flatten(0, 1),     # [B*N]
        reduction="mean"
    )

    # 回归：L1（演示版）
    det_loss_box = F.l1_loss(pred_boxes, gt_boxes, reduction="mean")

    loss_det = det_loss_cls + det_loss_box

    # -------------------------
    # 2) Semantic alignment loss (L_align)
    # -------------------------
    # 这里用 PDT 的 logits 和 pseudo target t̃_n
    # 为简单起见，用 labels 当作 target（实际应由 g_φ 生成 soft target）
    align_loss_ce = F.cross_entropy(
        cls_pdt.flatten(0, 1),
        labels.flatten(0, 1),
        reduction="mean"
    )

    loss_align = align_loss_ce + loss_kl

    # -------------------------
    # 3) Dual-stream gradient isolation (PD-DuGi)
    # -------------------------
    # 假设你给 detection 分支和 semantic 分支分别配了 optimizer
    # 比如：
    #   det_params  = backbone + decoder + bbox_head + det_cls_head
    #   sem_params  = proj_head + semantic_enhancer + pdt_head

    # (1) 更新 detection 分支（不回传到 semantic）
    if optimizer_det is not None:
        optimizer_det.zero_grad()
        loss_det.backward(retain_graph=True)
        optimizer_det.step()

    # (2) 更新 semantic 分支（不回传到 detection）
    if optimizer_sem is not None:
        optimizer_sem.zero_grad()
        # 这里最简单粗暴的方式：对 detection 分支做 detach
        # 实际实现时可以在构建 optimizer 时就不包含 detection 分支参数，
        # 这样 loss_align.backward() 的梯度只会流向 semantic 分支。
        loss_align.backward()
        optimizer_sem.step()

    return {
        "loss_det": loss_det.item(),
        "loss_align": loss_align.item(),
        "loss_kl": loss_kl.item() if isinstance(loss_kl, Tensor) else float(loss_kl),
    }



def test_dynamic_prototype_bank():
    torch.manual_seed(42)

    # ============================
    # 1. 创建一个 prototype bank
    # ============================
    num_prototypes = 10
    dim = 32
    bank = DynamicPrototypeBank(num_prototypes, dim, momentum=0.9)

    print("Prototype shape:", bank.prototypes.shape)  # [M, D]

    # ============================
    # 2. 准备“输入特征” e_i 用于 momentum update
    # ============================
    N = 5  # 几个 region feature
    feats = torch.randn(N, dim)

    print("\nBefore update:")
    print(bank.prototypes[:2])  # 打印前两个 prototype

    # 执行 momentum 更新
    bank.momentum_update(feats)

    print("\nAfter update:")
    print(bank.prototypes[:2])

    # ============================
    # 3. 测试 forward (query → prototype softmax)
    # ============================
    B, Q = 2, 3  # batch=2, query=3
    queries = torch.randn(B, Q, dim)

    weights = bank(queries)
    print("\nWeights shape:", weights.shape)
    print("Weights sum over M (should be 1):", weights[0, 0].sum())

    # ====== 再加一个断点 ======
    import ipdb; ipdb.set_trace()


if __name__ == "__main__":
    test_dynamic_prototype_bank()



