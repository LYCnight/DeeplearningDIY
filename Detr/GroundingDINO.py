import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

# ------------------------------------------------------------
# Focal Loss for binary logits (multi-label classification)
# ------------------------------------------------------------
class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        """
        logits:  Tensor[Q, T]
                 Q = num_queries
                 T = num_tokens (文本 token 数，例如 BERT 输出的词数)
                 
        targets: Tensor[Q, T]
                 每个 query 对应每个 token 的 0/1 匹配标签
        """

        # ---------------------------
        # prob: [Q, T]  每个 query 对 token 的概率
        # ---------------------------
        prob = torch.sigmoid(logits)

        # ---------------------------
        # ce_loss: [Q, T]
        # 每个 (query, token) 的 BCE loss
        # ---------------------------
        ce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )

        # ---------------------------
        # p_t: [Q, T]
        # p_t = prob   (if target=1)
        # p_t = 1-prob (if target=0)
        # focal loss 的核心 pt
        # ---------------------------
        p_t = prob * targets + (1 - prob) * (1 - targets)

        # ---------------------------
        # loss: [Q, T]
        # focal modulating factor (1 - p_t)^gamma
        # ---------------------------
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        # ---------------------------
        # alpha_t: [Q, T]
        # 正/负类的 alpha 平衡因子
        # ---------------------------
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # ---------------------------
        # loss: [Q, T]
        # 最终加入 alpha 权重
        # ---------------------------
        loss = alpha_t * loss

        # ---------------------------
        # mean over all queries & tokens
        # scalar
        # ---------------------------
        return loss.mean()



# ------------------------------------------------------------
# GIoU Loss
# ------------------------------------------------------------
def giou_loss(pred, target):
    """
    pred, target: [N, 4] in xyxy format
    """
    x1 = torch.max(pred[:, 0], target[:, 0])
    y1 = torch.max(pred[:, 1], target[:, 1])
    x2 = torch.min(pred[:, 2], target[:, 2])
    y2 = torch.min(pred[:, 3], target[:, 3])

    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)

    area_p = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    area_g = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])

    union = area_p + area_g - inter
    iou = inter / union.clamp(min=1e-6)

    # enclosing box
    cx1 = torch.min(pred[:, 0], target[:, 0])
    cy1 = torch.min(pred[:, 1], target[:, 1])
    cx2 = torch.max(pred[:, 2], target[:, 2])
    cy2 = torch.max(pred[:, 3], target[:, 3])

    C = (cx2 - cx1) * (cy2 - cy1)
    giou = iou - (C - union) / C.clamp(min=1e-6)

    return (1 - giou).mean()


# ------------------------------------------------------------
# Hungarian matcher
# ------------------------------------------------------------
class HungarianMatcher(nn.Module):
    def __init__(self, cls_weight=2.0, l1_weight=5.0, giou_weight=2.0):
        super().__init__()
        self.cls_weight = cls_weight
        self.l1_weight = l1_weight
        self.giou_weight = giou_weight

    def forward(self, logits, pred_boxes, tgt_tokens, tgt_boxes):
        """
        logits: [Q, T]
        pred_boxes: [Q, 4]
        tgt_tokens: [M, T]  # one-hot rows
        tgt_boxes:  [M, 4]
        """

        Q = logits.size(0)
        M = tgt_boxes.size(0)

        # classification cost = -logits for matched tokens
        cost_cls = -logits[:, tgt_tokens.argmax(dim=1)]  # [Q, M]

        # bbox L1 cost
        cost_l1 = torch.cdist(pred_boxes, tgt_boxes, p=1)  # [Q, M]

        # giou cost
        def pair_giou(p, g):
            p = p.unsqueeze(1).expand(Q, M, 4)
            g = g.unsqueeze(0).expand(Q, M, 4)

            x1 = torch.max(p[..., 0], g[..., 0])
            y1 = torch.max(p[..., 1], g[..., 1])
            x2 = torch.min(p[..., 2], g[..., 2])
            y2 = torch.min(p[..., 3], g[..., 3])

            inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
            area_p = (p[..., 2] - p[..., 0]) * (p[..., 3] - p[..., 1])
            area_g = (g[..., 2] - g[..., 0]) * (g[..., 3] - g[..., 1])
            union = area_p + area_g - inter

            iou = inter / union.clamp(min=1e-6)

            cx1 = torch.min(p[..., 0], g[..., 0])
            cy1 = torch.min(p[..., 1], g[..., 1])
            cx2 = torch.max(p[..., 2], g[..., 2])
            cy2 = torch.max(p[..., 3], g[..., 3])

            C = (cx2 - cx1) * (cy1 - cy1)
            giou = iou - (C - union) / C.clamp(min=1e-6)
            return 1 - giou

        cost_giou = pair_giou(pred_boxes, tgt_boxes)

        # combined matching cost
        C = (
            self.cls_weight * cost_cls +
            self.l1_weight * cost_l1 +
            self.giou_weight * cost_giou
        )  # [Q, M]

        C = C.cpu()
        indices = linear_sum_assignment(C)  # Hungarian
        return indices


# ------------------------------------------------------------
# Final GroundingDINO Loss
# ------------------------------------------------------------
class GroundingDINOLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.matcher = HungarianMatcher()
        self.focal = BinaryFocalLoss()

    def forward(self, logits, pred_boxes, tgt_tokens, tgt_boxes):
        """
        logits: [Q, T]
        pred_boxes: [Q, 4]
        tgt_tokens: [M, T]   (one hot token assignment)
        tgt_boxes:  [M, 4]
        """

        # --- 1. Hungarian matching ---
        idx_q, idx_tgt = self.matcher(logits, pred_boxes, tgt_tokens, tgt_boxes)

        # --- 2. Build matched target for classification ---
        T = logits.size(1)
        cls_target = torch.zeros_like(logits)
        cls_target[idx_q] = tgt_tokens[idx_tgt]  # place matched tokens

        # --- 3. Classification loss (Focal Loss) ---
        loss_cls = self.focal(logits, cls_target)

        # --- 4. Box regression on matched pairs ---
        matched_pred = pred_boxes[idx_q]
        matched_gt = tgt_boxes[idx_tgt]

        loss_l1 = F.l1_loss(matched_pred, matched_gt)
        loss_giou = giou_loss(matched_pred, matched_gt)

        # final loss
        total = loss_cls + 5.0 * loss_l1 + 2.0 * loss_giou
        return total, loss_cls, loss_l1, loss_giou



