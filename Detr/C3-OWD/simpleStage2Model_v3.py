# ==========================================================
# Fake C3-OWD Stage2 — Minimal, Standalone, Runnable Version
# ==========================================================
# 不依赖 mmdet / mmcv / 你的工程结构
# 可直接 python fake_codetr_stage2.py
# ==========================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
import copy


# ----------------------------------------------------------
# Momentum encoder（简化版，忠实还原 MoCo 原理）
# ----------------------------------------------------------
class MomentumEncoder(nn.Module):
    def __init__(self, encoder, momentum=0.999):
        super().__init__()
        self.encoder = encoder
        self.momentum = momentum
        self.encoder_m = copy.deepcopy(encoder)

        for p in self.encoder_m.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def _update_m(self):
        for p, pm in zip(self.encoder.parameters(), self.encoder_m.parameters()):
            pm.data = pm.data * self.momentum + p.data * (1 - self.momentum)

    def forward(self, x, use_momentum=False):
        if use_momentum:
            self._update_m()
            with torch.no_grad():
                return self.encoder_m(x)
        return self.encoder(x)


# ----------------------------------------------------------
# Fake backbone (代替 ResNet)
# 输出模拟 256 channels, 32×32
# ----------------------------------------------------------
class FakeBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 256, kernel_size=3, padding=1)

    def forward(self, x):
        # 输入 (B,3,256,256) → 输出 (B,256,32,32)
        x = self.conv(x)        # /1   (256 -> 256)
        x = F.avg_pool2d(x, 8)  # /8    (256 -> 32)
        return x     # (B,256,32,32)


# ----------------------------------------------------------
# Fake neck（模拟 FPN 结构）
# ----------------------------------------------------------
class FakeNeck(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln = nn.LayerNorm(256)

    def forward(self, x):
        # 输入 (B,256,32,32) → 输出 (1, B,256,32,32)
        # 原版 neck 会处理多个层，这里只处理一个
        B, C, H, W = x.shape
        x = x.view(B, C, -1).transpose(1, 2)    # (B, HW, C)
        x = self.ln(x)
        x = x.transpose(1, 2).view(B, C, H, W)   # (B, C, H, W)
        return (x,) 



# ----------------------------------------------------------
# Fake TFB (代替 RWKV 融合)
# ----------------------------------------------------------
class FakeTFB(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_fuse = nn.Conv2d(256, 256, 1)

    def forward(self, x_vis, x_lwir):
        # 2 * (B,256,32,32) → (B,256,32,32)
        return [self.conv_fuse(x_vis + x_lwir)]


# ----------------------------------------------------------
# Fake DETR Query Head
# 输出伪 bbox loss
# ----------------------------------------------------------
# class FakeQueryHead(nn.Module):
#     def __init__(self, num_classes):
#         super().__init__()
#         self.fc = nn.Linear(256, num_classes)  

#     def forward_train(self, feats, img_metas, gt_bboxes, gt_labels, _):
#         # feats: tuple([tensor])  # (1, B, C, H, W)
#         import ipdb; ipdb.set_trace()
#         x = feats[0]    # [4, 256, 32, 32]  # (B,256,32,32)
#         B, C, H, W = x.shape
#         pooled = F.adaptive_avg_pool2d(x, 1).view(B, C)   # (B, C)
#         logits = self.fc(pooled)  # (B, class)  # (4, 3)
#         all_labels = torch.cat(gt_labels, dim=0)  # (n_bbox,)
#         loss = F.cross_entropy(logits, all_labels)
#         return {"loss_query": loss}, feats

class FakeQueryHead(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.fc = nn.Linear(256, num_classes)

    def forward_train(self, feats, img_metas, gt_bboxes, gt_labels, _):
        """
        feats[0]: (B,256,32,32)
        gt_labels: list[tensor], 每张图的目标数不同
        """
        x = feats[0]     # (B,256,32,32)
        B, C, H, W = x.shape

        # 平均池化图像级 feature → (B,256)
        pooled = F.adaptive_avg_pool2d(x, 1).view(B, C)   # (B,256)

        # -----------------------------------------------------
        # 对每张图按 bbox 数重复 pooled 特征：
        # image0 有2个框 → [pool0, pool0]
        # image1 有1个框 → [pool1]
        # ...
        # -----------------------------------------------------
        pooled_list = []
        label_list = []

        for i in range(B):
            num_obj = len(gt_labels[i])      # 当前图多少目标
            pooled_list.append(pooled[i].unsqueeze(0).repeat(num_obj, 1))  # (num_obj,256)
            label_list.append(gt_labels[i])  # (num_obj,)

        pooled_all = torch.cat(pooled_list, dim=0)    # (sum(num_obj), 256)
        labels_all = torch.cat(label_list, dim=0)     # (sum(num_obj),)

        logits = self.fc(pooled_all)   # (sum(num_obj), num_classes)

        loss = F.cross_entropy(logits, labels_all)

        return {"loss_query": loss}, feats



# ----------------------------------------------------------
# Gaussian OVOD loss（简化原版）
# ----------------------------------------------------------
def gaussian_focal_loss(pred, gt_bboxes, gt_labels, img_metas, num_classes):
    '''
    pred: (B, n_class, H, W)
    gt_bboxes: (B, nclass, 4)
    gt_labels: (B, nclass)
    '''
    b, n_class, h, w = pred.shape
    device = pred.device

    target = torch.zeros_like(pred)

    # 遍历 batch
    for i, (boxes, labels) in enumerate(zip(gt_bboxes, gt_labels)):
        # 缩放比例
        img_h, img_w = img_metas[0]
        sx = w / img_w
        sy = h / img_h

        # 遍历 bbox
        for box, cls in zip(boxes, labels):
            x1, y1, x2, y2 = box
            x1 *= sx; x2 *= sx
            y1 *= sy; y2 *= sy

            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            sigma_x = max(1.0, (x2 - x1) / 6)
            sigma_y = max(1.0, (y2 - y1) / 6)

            yy, xx = torch.meshgrid(
                torch.arange(h, device=device),
                torch.arange(w, device=device),
                indexing="ij"
            )

            g = torch.exp(-((xx - cx) ** 2 / (2 * sigma_x ** 2) +
                            (yy - cy) ** 2 / (2 * sigma_y ** 2)))
            target[i, cls] = torch.maximum(target[i, cls], g)

    prob = torch.sigmoid(pred)
    pt = torch.where(target > 0, prob, 1 - prob)
    return (-torch.log(pt + 1e-6)).mean()



# ----------------------------------------------------------
# 多模板 CLIP 文本增强   prompt ensembling
# ----------------------------------------------------------
def build_text_features(classes, device):
    """
    输入：类别名称 list，例如 ["person", "car", "bus"]
    输出：text_features: (num_classes, 512)
    —— prompt ensemble + L2 norm
    """

    # 加载 CLIP
    clip_model, _ = clip.load("ViT-B/32", device=device)
    clip_model.eval()

    # 多模板 prompt（增强版）
    templates = [
        "a photo of a {}",
        "an image of a {}",
        "a {} in the scene",
        "there is a {} in this image"
    ]

    all_text_features = []

    with torch.no_grad():
        for template in templates:
            texts = [template.format(c) for c in classes]
            tokens = clip.tokenize(texts).to(device)

            feat = clip_model.encode_text(tokens)           # (num_classes, 512)
            feat = feat / feat.norm(dim=-1, keepdim=True)   # normalize

            all_text_features.append(feat)

    # 对 prompt 取平均（prompt ensemble）
    all_text_features = torch.stack(all_text_features, dim=0)   # (T, num_classes, 512)
    text_features = all_text_features.mean(dim=0)               # (num_classes, 512)

    # 再归一化一次（CLIP 规范）
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return text_features

# ----------------------------------------------------------
# TwoStreamCoDETR Fake 版本（可运行）
# ----------------------------------------------------------
class TwoStreamCoDETR_Fake(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.classes = classes
        self.num_classes = len(classes)
        self.vison_feat_proj = nn.Linear(256, 768)

        self.backbone_vis = FakeBackbone()
        self.backbone_lwir = FakeBackbone()

        self.momentum_vis = MomentumEncoder(self.backbone_vis)
        self.momentum_lwir = MomentumEncoder(self.backbone_lwir)

        self.tfb = FakeTFB()
        self.neck = FakeNeck()
        self.query_head = FakeQueryHead(self.num_classes)

        self.clip_model = None
        self.text_feats = None

    def _init_clip(self, device):
        # self.text_feats: (num_classes, 512)
        if self.clip_model is None:
            self.clip_model, _ = clip.load("ViT-B/32", device=device)
            self.clip_model.eval()
            self.text_feats = build_text_features(self.classes, device)

    # ---------------- forward_train -------------------
    def forward_train(self, img, img_lwir, gt_bboxes, gt_labels):

        # backbone
        f_vis = self.backbone_vis(img)
        f_lwir = self.backbone_lwir(img_lwir)

        # momentum encoder
        with torch.no_grad():
            f_vis_m = self.momentum_vis(img, use_momentum=True)
            f_lwir_m = self.momentum_lwir(img_lwir, use_momentum=True)

        # fusion
        feats = self.tfb(f_vis, f_lwir)
        feats = self.neck(feats[0])   # (1,B,C,H,W) -> tuple   # [1, 1, 256, 32, 32]

        # CLIP OVOD logits
        feat = feats[0]   # (B,C,H,W)
        B, C, H, W = feat.shape
        feat_flat = feat.view(B, C, -1).permute(0, 2, 1)   # (1, C, HW) -> (1, HW, C)  # (1, 1024, 256)
        device = feat.device
        self._init_clip(device)

        # 用 CLIP projection
        with torch.no_grad():
            proj = self.clip_model.visual.proj   # (768, 512)
        feat_768 = self.vison_feat_proj(feat_flat)  # (1, 1024, 256) -> (1, 1024, 768)
        feat_512 = feat_768 @ proj          #  (1, 1024, 768) -> # (1, 1024, 512)
        feat_512 = feat_512 / feat_512.norm(dim=-1, keepdim=True)   # (1, 1024, 512)

        # feat_512: (b, HW, 512)
        # self.text_feats: (n_class, 512)
        logits = torch.einsum("bhc,kc->bhk", feat_512, self.text_feats)   # (B, HW, n_class)  # [1, 1024, 3]
        logits = logits.permute(0, 2, 1).view(B, self.num_classes, H, W)  # (B, n_class, HW) -> (1, n_class, H, W)  # (1, 3, 32, 32)

        # simple OVOD loss
        # img.shape: (1, 3, 256, 256)
        img_meta = [(img.shape[2], img.shape[3])]
        # img_meta = [(img[i].shape[2], img[i].shape[3]) for i in range(B)]

        ovod_loss = gaussian_focal_loss(logits, gt_bboxes, gt_labels, img_meta, self.num_classes)

        # query head loss
        # feats: # (1, 4, 256, 32, 32)   # (1, B, C, H, W)
        # img_meta: (2,
        # gt_bboxes: (4, 4, 4)  # (B, n_bbox, 4)
        # gt_labels: (4, 4)  # (B, n_bbox)
        det_loss, _ = self.query_head.forward_train(feats, img_meta, gt_bboxes, gt_labels, None)

        return {
            "loss_ovod": ovod_loss,
            **det_loss
        }


# ==========================================================
#                 测试入口
# ==========================================================
if __name__ == "__main__":
    print("🔥 Running Fake C3-OWD Stage2 Model...")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 类别定义
    object_classes = ["person", "car", "bus"]
    
    # model_init
    model = TwoStreamCoDETR_Fake(classes=object_classes).to(device)

    # Fake inputs
    img = torch.randn(4, 3, 256, 256).to(device)
    img_lwir = torch.randn(4, 3, 256, 256).to(device)



    # bbox 
    gt_bboxes = [
        # image 0：2 个框
        torch.tensor([
            [50, 60, 180, 200],     # box A
            [120, 30, 210, 140],    # box B
        ], dtype=torch.float32).to(device),

        # image 1：1 个框
        torch.tensor([
            [30, 40, 160, 180],
        ], dtype=torch.float32).to(device),

        # image 2：2 个框
        torch.tensor([
            [10, 20, 80, 120],
            [140, 100, 240, 230],
        ], dtype=torch.float32).to(device),

        # image 3：1 个框
        torch.tensor([
            [60, 70, 200, 240],
        ], dtype=torch.float32).to(device),
    ]

    # labels
    gt_labels = [
        torch.tensor([1, 0], dtype=torch.long).to(device),   # image 0 → car, person
        torch.tensor([2], dtype=torch.long).to(device),      # image 1 → bus
        torch.tensor([0, 1], dtype=torch.long).to(device),   # image 2 → person, car
        torch.tensor([1], dtype=torch.long).to(device),      # image 3 → car
    ]

    # loss
    losses = model.forward_train(img, img_lwir, gt_bboxes, gt_labels)

    print("💡 Losses:")
    for k, v in losses.items():
        print(f"{k}: {float(v):.6f}")

    print("\n🎉 Fake C3-OWD Stage2 test completed!")
