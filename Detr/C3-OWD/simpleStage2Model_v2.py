# ============================================================
# Simplified V2: 加入原版关键增强内容（已精炼）
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import copy


# ------------------------------------------------------------
# 1) Tiny Backbone（不变）
# ------------------------------------------------------------
class TinyBackbone(nn.Module):
    def __init__(self, in_ch=3, feat_ch=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, feat_ch, 3, 2, 1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


# ------------------------------------------------------------
# 2) Momentum Encoder（不变）
# ------------------------------------------------------------
class MomentumEncoder(nn.Module):
    def __init__(self, encoder, m=0.999):
        super().__init__()
        self.encoder = encoder
        self.m = m
        self.encoder_m = copy.deepcopy(encoder)
        for p in self.encoder_m.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def _update_m(self):
        for q, k in zip(self.encoder.parameters(), self.encoder_m.parameters()):
            k.data = k.data * self.m + q.data * (1 - self.m)

    def forward(self, x, use_momentum=False):
        if use_momentum:
            with torch.no_grad():
                self._update_m()
                return self.encoder_m(x)
        return self.encoder(x)


# ------------------------------------------------------------
# 3) Feature Bank（加入 cap + hard-negative）
# ------------------------------------------------------------
class FeatureBank:
    def __init__(self, bank_size=4096):
        self.features = deque(maxlen=bank_size)
        self.labels = deque(maxlen=bank_size)

    def update(self, feats, labels):
        feats = feats.detach().cpu()
        labels = labels.detach().cpu()
        for f, y in zip(feats, labels):
            self.features.append(f)
            self.labels.append(y)

    def get(self, device):
        if len(self.features) == 0:
            return None, None
        f = torch.stack(list(self.features)).to(device)
        y = torch.stack(list(self.labels)).to(device)
        return f, y


# ------------------------------------------------------------
# 4) Adaptive Gaussian（来自原版精简）
# ------------------------------------------------------------
def adaptive_gaussian(h, w, cx, cy, bw, bh, device):
    y, x = torch.meshgrid(
        torch.arange(h, device=device),
        torch.arange(w, device=device),
        indexing="ij"
    )

    # 原版：基于 bbox area 的自适应 sigma
    sigma_x = max(1.0, (bw / 6) * (bw * bh / (h * w)).pow(0.3))
    sigma_y = max(1.0, (bh / 6) * (bw * bh / (h * w)).pow(0.3))

    g = torch.exp(-((x - cx)**2 / (2 * sigma_x**2)
                    + (y - cy)**2 / (2 * sigma_y**2)))

    # 加一点 broad gaussian（原版）
    sigma_x_b = sigma_x * 1.5
    sigma_y_b = sigma_y * 1.5
    g_b = torch.exp(-((x - cx)**2 / (2 * sigma_x_b**2)
                      + (y - cy)**2 / (2 * sigma_y_b**2)))

    return torch.maximum(g, 0.3 * g_b)   # soft blend


# ------------------------------------------------------------
# 5) Build Gaussian Target（加入 adaptive + blend）
# ------------------------------------------------------------
def build_gaussian_target_v2(logits, gt_bboxes, gt_labels, img_sizes, num_classes):
    B, C, H, W = logits.shape
    device = logits.device
    target = torch.zeros(B, C, H, W, device=device)

    for b in range(B):
        boxes = gt_bboxes[b]
        labels = gt_labels[b]
        if len(boxes) == 0:
            continue

        H_img, W_img = img_sizes[b]
        scale_h, scale_w = H / H_img, W / W_img

        boxes_s = boxes.clone()
        boxes_s[:, [0, 2]] *= scale_w
        boxes_s[:, [1, 3]] *= scale_h

        for box, cls in zip(boxes_s, labels):
            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2)/2, (y1 + y2)/2
            bw, bh = (x2 - x1), (y2 - y1)

            g = adaptive_gaussian(H, W, cx, cy, bw, bh, device)

            # soft overlapping blend（来自原版）
            target[b, cls] = torch.maximum(target[b, cls], g)

    return target


# ------------------------------------------------------------
# 6) Gaussian Focal Loss（加入 multi-scale consistency）
# ------------------------------------------------------------
def gaussian_focal_loss_v2(logits, target, alpha=0.25, gamma=2.0):
    prob = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")

    pos = target > 0.1
    pt = torch.where(pos, prob, 1 - prob)
    alpha_t = torch.where(pos, alpha, 1 - alpha)

    focal_weight = alpha_t * (1 - pt).pow(gamma)
    main = (focal_weight * bce).mean()

    # 原版：multi-scale consistency
    if logits.shape[-1] >= 8:
        p2 = F.avg_pool2d(prob, 2)
        t2 = F.avg_pool2d(target, 2)
        main += 0.1 * F.mse_loss(p2, t2)

    return main


# ------------------------------------------------------------
# 7) Contrastive Loss（加入 hard negative mining）
# ------------------------------------------------------------
def contrastive_loss_v2(q, labels_q, bank: FeatureBank, temperature=0.07, hard_ratio=0.3):
    device = q.device
    q = F.normalize(q, dim=1)

    bank_f, bank_y = bank.get(device)
    if bank_f is None:
        return torch.tensor(0.0, device=device)

    bank_f = F.normalize(bank_f, dim=1)

    sim = q @ bank_f.t() / temperature  # (B,N)

    # positive mask
    pos_mask = (labels_q.view(-1,1) == bank_y.view(1,-1))

    if pos_mask.sum() == 0:
        return torch.tensor(0.0, device=device)

    # hard negative mining
    neg_mask = ~pos_mask
    neg_scores = sim.masked_select(neg_mask)
    k = max(1, int(len(neg_scores) * hard_ratio))
    topk_neg = neg_scores.topk(k).values.mean()

    # log-softmax InfoNCE
    log_prob = sim.log_softmax(dim=1)
    pos_log_prob = (log_prob * pos_mask).sum(dim=1) / (pos_mask.sum(dim=1)+1e-6)

    loss = -(pos_log_prob.mean() - topk_neg * 0.05)  # push positives, pull hard negs
    return loss


# ------------------------------------------------------------
# 8) 简化版 Stage2 主模型（加入 region pooling）
# ------------------------------------------------------------
class SimpleStage2V2(nn.Module):
    def __init__(self, num_classes, feat_dim=256, text_dim=512):
        super().__init__()
        self.num_classes = num_classes

        self.backbone = TinyBackbone(3, feat_dim)
        self.momentum_backbone = MomentumEncoder(self.backbone, 0.999)

        # projection for OVOD
        self.visual_proj = nn.Linear(feat_dim, text_dim)

        # fake CLIP text features
        text = torch.randn(num_classes, text_dim)
        self.register_buffer("text_feats", F.normalize(text, dim=1))

        # contrastive
        self.contrast_proj = nn.Linear(feat_dim, feat_dim)
        self.bank = FeatureBank()

    # --------------------------
    # Region pooling = ROIAlign 简化
    # --------------------------
    def region_pool(self, feat, x1,y1,x2,y2, out=4):
        patch = feat[:, :, y1:y2, x1:x2]  # (1,C,h,w)
        return F.adaptive_avg_pool2d(patch, (out,out)).mean(dim=[2,3])  # (1,C)

    # --------------------------
    def forward_train(self, img, gt_boxes, gt_labels, img_sizes):
        B = img.size(0)
        device = img.device

        feat = self.backbone(img)      # (B,256,Hf,Wf)
        B,C,Hf,Wf = feat.shape

        # OVOD
        flat = feat.flatten(2).permute(0,2,1)
        emb = F.normalize(self.visual_proj(flat), dim=-1)
        logits = emb @ self.text_feats.t()  # (B,HW,class)
        logits = logits.permute(0,2,1).reshape(B,self.num_classes,Hf,Wf)

        # Gaussian target V2
        target = build_gaussian_target_v2(logits, gt_boxes, gt_labels, img_sizes, self.num_classes)
        ovod_loss = gaussian_focal_loss_v2(logits, target)

        # Momentum feature
        with torch.no_grad():
            feat_m = self.momentum_backbone(img, use_momentum=True)

        # Region q/k extraction
        q_list, y_list, k_list = [], [], []
        for b in range(B):
            boxes = gt_boxes[b]
            labels = gt_labels[b]
            if len(boxes)==0: continue

            H_img,W_img = img_sizes[b]
            boxes_s = boxes.clone()
            boxes_s[:,[0,2]] *= (Wf/W_img)
            boxes_s[:,[1,3]] *= (Hf/H_img)

            for box,cls in zip(boxes_s, labels):
                x1,y1,x2,y2 = box.int()
                x1,x2 = x1.clamp(0,Wf-1), x2.clamp(0,Wf-1)
                y1,y2 = y1.clamp(0,Hf-1), y2.clamp(0,Hf-1)
                if x2<=x1 or y2<=y1: continue

                q = self.region_pool(feat[b:b+1], x1,y1,x2,y2) # (1,C)
                k = self.region_pool(feat_m[b:b+1], x1,y1,x2,y2)

                q_list.append(q)
                k_list.append(k)
                y_list.append(cls.view(1))

        # contrastive
        if q_list:
            q = torch.cat(q_list,0)
            q = self.contrast_proj(q)
            y = torch.cat(y_list,0)
            k = torch.cat(k_list,0)

            self.bank.update(k, y)
            cont_loss = contrastive_loss_v2(q, y, self.bank)
        else:
            cont_loss = torch.tensor(0.0, device=device)

        return {
            "ovod_loss": ovod_loss,
            "contrast_loss": cont_loss,
        }
