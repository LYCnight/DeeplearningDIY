import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import copy


# -----------------------
# 1. 简单 backbone
# -----------------------
class TinyBackbone(nn.Module):
    """很小的 CNN，当成视觉 backbone 用"""
    def __init__(self, in_ch=3, feat_ch=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, 2, 1),  # /2
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1),    # /4
            nn.ReLU(inplace=True),
            nn.Conv2d(128, feat_ch, 3, 2, 1),  # /8
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)   # (B, 256, H/8, W/8)


# -----------------------
# 2. 动量 encoder（MoCo 核心）
# -----------------------
class MomentumEncoder(nn.Module):
    def __init__(self, encoder, m=0.999):
        super().__init__()
        self.encoder = encoder
        self.m = m
        self.encoder_m = copy.deepcopy(encoder)
        for p in self.encoder_m.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def _update_momentum(self):
        for p_q, p_k in zip(self.encoder.parameters(), self.encoder_m.parameters()):
            p_k.data = p_k.data * self.m + p_q.data * (1 - self.m)

    def forward(self, x, use_momentum=False):
        if use_momentum:
            with torch.no_grad():
                self._update_momentum()
                return self.encoder_m(x)
        else:
            return self.encoder(x)


# -----------------------
# 3. 简易 FeatureBank（队列）
# -----------------------
class FeatureBank:
    def __init__(self, bank_size=4096):
        self.features = deque(maxlen=bank_size)
        self.labels = deque(maxlen=bank_size)

    def update(self, feats, labels):
        # feats: (B, D); labels: (B,)
        feats = feats.detach().cpu()
        labels = labels.detach().cpu()
        for f, y in zip(feats, labels):
            self.features.append(f)
            self.labels.append(y)

    def get_all(self, device):
        if len(self.features) == 0:
            return None, None
        feats = torch.stack(list(self.features)).to(device)   # (N, D)
        labels = torch.stack(list(self.labels)).to(device)    # (N,)
        return feats, labels


# -----------------------
# 4. 高斯 Target + Focal Loss（简化版）
# -----------------------
def gaussian_2d(h, w, cx, cy, bw, bh, device):
    y, x = torch.meshgrid(
        torch.arange(h, device=device),
        torch.arange(w, device=device),
        indexing='ij'
    )
    sigma_x = max(1.0, bw / 6)
    sigma_y = max(1.0, bh / 6)
    g = torch.exp(-((x - cx)**2 / (2 * sigma_x**2) +
                    (y - cy)**2 / (2 * sigma_y**2)))
    return g


def build_gaussian_target(logits, gt_bboxes, gt_labels, img_sizes, num_classes):
    """
    logits: (B, C, H, W)
    gt_bboxes: List[Tensor(ni,4)]  x1,y1,x2,y2 in img coords   # (B, nbbox, 4)
    gt_labels: List[Tensor(ni,)]                               # (B, nlabels)
    img_sizes: List[(H_img, W_img)]                            # (B, 1, 1)
    """
    B, C, H, W = logits.shape
    device = logits.device
    target = torch.zeros(B, C, H, W, device=device)

    # 遍历 batches
    for b in range(B):
        bboxes = gt_bboxes[b]   # (nbbox, 4)
        labels = gt_labels[b]   # (nlabels, )
        if bboxes.numel() == 0:
            continue
        H_img, W_img = img_sizes[b]
        scale_h, scale_w = H / H_img, W / W_img

        bboxes_scaled = bboxes.clone()

        # (nbbox, 2)
        bboxes_scaled[:, [0, 2]] *= scale_w   # x 坐标
        # (nbbox, 2)
        bboxes_scaled[:, [1, 3]] *= scale_h   # y 坐标

        # 遍历所有 bbox
        for box, cls in zip(bboxes_scaled, labels):
            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            bw, bh = (x2 - x1), (y2 - y1)
            g = gaussian_2d(H, W, cx, cy, bw, bh, device)
            # target[b, cls]: (H, W);    g: (H, W)
            target[b, cls] = torch.maximum(target[b, cls], g)

    return target  # (B, C, H, W)


def gaussian_focal_loss(logits, target, alpha=0.25, gamma=2.0):
    """
    logits & target: (B, C, H, W)
    """
    prob = torch.sigmoid(logits)  # logits.shape: (2, 3, 32, 32)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
    pos_mask = target > 0.1
    pt = torch.where(pos_mask, prob, 1 - prob)
    alpha_t = torch.where(pos_mask, alpha, 1 - alpha)
    focal_weight = alpha_t * (1 - pt) ** gamma
    loss = (focal_weight * bce).mean()
    return loss


# -----------------------
# 5. 简化版 InfoNCE 对比损失
# -----------------------
def contrastive_loss_moco(q_feat, q_label, bank: FeatureBank, temperature=0.07):
    """
    q_feat:  (B, D) 当前 batch 的 region 特征
    q_label:(B,)
    bank:    动量队列，里面是 (N, D)
    """
    device = q_feat.device
    q_feat = F.normalize(q_feat, dim=1)  # (B, D)

    bank_feats, bank_labels = bank.get_all(device)
    if bank_feats is None:
        # 队列还没填满，用简单正样本相似度
        return torch.tensor(0.0, device=device)

    bank_feats = F.normalize(bank_feats, dim=1)  # (N, D)

    # 相似度: (B, N)
    sim = q_feat @ bank_feats.t() / temperature  # (B, N)

    # 正样本：label 相同的那些 bank index
    # 对 InfoNCE 来说，我们可以简单地：对每个 q，从 bank 里找所有同类为正，其余为负
    import ipdb; ipdb.set_trace()
    # q_label:  # (B,) 
    # q_label.unsqueeze(1)  # (B, 1)      
    # bank_labels.unsqueeze(0)  # (1, N)
    pos_mask = (q_label.unsqueeze(1) == bank_labels.unsqueeze(0))  # (B,N)

    # 避免没有正样本（极端情况）
    if pos_mask.sum() == 0:
        return torch.tensor(0.0, device=device)

    # log-softmax 形式的 InfoNCE
    log_prob = sim.log_softmax(dim=1)  # (B,N)

    # 只在正样本位置取平均
    pos_log_prob = (log_prob * pos_mask.float()).sum(dim=1) / (pos_mask.float().sum(dim=1) + 1e-6)
    loss = -pos_log_prob.mean()
    return loss


# -----------------------
# 6. 一个极简的 Stage2 模型
# -----------------------
class SimpleStage2Model(nn.Module):
    def __init__(self, num_classes, feat_dim=256, text_dim=512):
        super().__init__()
        self.num_classes = num_classes

        # 视觉 backbone + 动量分支
        self.backbone = TinyBackbone(in_ch=3, feat_ch=feat_dim)
        self.momentum_backbone = MomentumEncoder(self.backbone, m=0.999)

        # 把视觉特征投到 CLIP 维度，方便和文本相似度
        self.visual_proj = nn.Linear(feat_dim, text_dim)

        # 这里用随机文本特征代替 CLIP，真实场景就是 encode_text
        text_feats = torch.randn(num_classes, text_dim)
        self.register_buffer("text_feats", F.normalize(text_feats, dim=1))

        # 对比学习投影头 & 队列
        self.contrast_proj = nn.Linear(feat_dim, feat_dim)
        self.feature_bank = FeatureBank(bank_size=4096)

    def forward_train(self, img, gt_bboxes, gt_labels, img_sizes):
        """
        img: (B,3,H,W)
        gt_bboxes: list[Tensor(ni,4)]   # (b, nbbox, 4)
        gt_labels: list[Tensor(ni,)]    # (b, nlabels)
        img_sizes: list[(H_img,W_img)]  # (b, 2)
        """
        B = img.size(0)
        device = img.device

        # 1) 视觉特征
        # img.shape: (2, 3, 256, 256)  # (B, class, H, W)
        # feat.shape: (2, 256, 32, 32)  # (B, C, Hf, Wf)
        feat = self.backbone(img)      # (B, C, Hf, Wf)  # ()
        B, C, Hf, Wf = feat.shape

        # 2) CLIP 相似度 → ovod_logits
        feat_flat = feat.flatten(2).permute(0, 2, 1)   # (B, HW, C)   # (2, 1024, 512)
        vis_emb = self.visual_proj(feat_flat)          # (B, HW, E)
        vis_emb = F.normalize(vis_emb, dim=-1)          # (2, 1024, 512)
        text_emb = self.text_feats                     # (class, E)   # (3, 512)

        # 相似度: (B, HW, num_classes)
        sim = vis_emb @ text_emb.t()                   # (B,HW,class)      # (2, 1024, 3)
        ovod_logits = sim.permute(0, 2, 1).reshape(B, self.num_classes, Hf, Wf)   # (B, class, HF, HW)  # (2, 3, 32, 32)


        # 3) 高斯 target + focal loss
        # ovod_logits: (B, C, HF, HW)  # (2, 3, 32, 32)
        # target: (B, C, HF, HW)  # (2, 3, 32, 32)
        target = build_gaussian_target(
            ovod_logits, gt_bboxes, gt_labels, img_sizes, self.num_classes
        )
        ovod_loss = gaussian_focal_loss(ovod_logits, target)  # (1,)

        # 4) MoCo 对比学习：用每个 bbox 的区域平均特征作为 q，动量分支作为 k-bank
        # 4.1 动量特征
        with torch.no_grad():
            feat_m = self.momentum_backbone(img, use_momentum=True)   # (B,C,Hf,Wf)

        region_q_list = []
        region_label_list = []
        region_k_list = []

        # 遍历 batches
        for b in range(B):
            boxes = gt_bboxes[b]
            labels = gt_labels[b]
            if boxes.numel() == 0:
                continue
            
            # 坐标缩放
            H_img, W_img = img_sizes[b]
            scale_h, scale_w = Hf / H_img, Wf / W_img
            boxes_scaled = boxes.clone()
            boxes_scaled[:, [0, 2]] *= scale_w
            boxes_scaled[:, [1, 3]] *= scale_h

            # 遍历 bbox
            for box, cls in zip(boxes_scaled, labels):
                x1, y1, x2, y2 = box.int()
                x1 = x1.clamp(0, Wf - 1)
                x2 = x2.clamp(0, Wf - 1)
                y1 = y1.clamp(0, Hf - 1)
                y2 = y2.clamp(0, Hf - 1)
                if x2 <= x1 or y2 <= y1:
                    continue
                
                # 当前分支的 region 平均特征
                # feat: (2, 256, 32, 32)   # (B, C, HF, HW)
                # feat[b:b+1, :, y1:y2, x1:x2]:   # (1, 256, 13, 9)  # (1, C, H_region, W_region)
                region_q = feat[b:b+1, :, y1:y2, x1:x2].mean(dim=[2, 3])  # (1,C)   # (1, 256)
                region_q_list.append(region_q)   # (nRegion, 1 , C)
                region_label_list.append(cls.view(1))    # (nRegion, 1)
                # cls: 标量   # cls.view(1): (1,)

                # 动量分支的 region 特征作为 k（放队列）
                region_k = feat_m[b:b+1, :, y1:y2, x1:x2].mean(dim=[2, 3])  # (1,C)
                region_k_list.append(region_k)   # (nRegion, 1 , C)

        if len(region_q_list) > 0:
            q = torch.cat(region_q_list, dim=0)               # (nRegion,C)  # (2, 256)
            q = self.contrast_proj(q)                         # (nRegion,C)  
            labels_q = torch.cat(region_label_list, dim=0)    # (nRegion,)

            k = torch.cat(region_k_list, dim=0)               # (nRegion,C)
            # 更新队列
            self.feature_bank.update(k, labels_q)

            cont_loss = contrastive_loss_moco(
                q, labels_q, self.feature_bank, temperature=0.07
            )
        else:
            cont_loss = torch.tensor(0.0, device=device)

        return {
            "ovod_loss": ovod_loss,
            "contrastive_loss": cont_loss
        }


# -----------------------
# 7. 小测试
# -----------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    model = SimpleStage2Model(num_classes=3)
    img = torch.randn(2, 3, 256, 256)   # 两张图

    gt_bboxes = [
        torch.tensor([[50, 60, 120, 160]], dtype=torch.float32),   # img0
        torch.tensor([[30, 40, 90, 140]], dtype=torch.float32)     # img1
    ]
    gt_labels = [
        torch.tensor([0]),  # 第一张类别 0
        torch.tensor([2])   # 第二张类别 2
    ]
    img_sizes = [(256, 256), (256, 256)]

    losses = model.forward_train(img, gt_bboxes, gt_labels, img_sizes)
    print({k: v.item() for k, v in losses.items()})
