'''
这是一个“基于高斯热图的目标检测损失 + Focal + 类别均衡 + 硬负样本挖掘 + 多尺度一致性”的完整 Loss Pipeline，用于训练一个 C3-OWD 风格的检测头。
'''


import torch
import torch.nn.functional as F

# ----------------------------------------------------------
# 1) 自适应二维高斯
# ----------------------------------------------------------
def gaussian_2d(h, w, cx, cy, bw, bh, adaptive=True, device="cpu"):
    # y: (H, 1) 广播为 H×W
    # x: (1, W) 广播为 H×W
    y, x = torch.meshgrid(
        torch.arange(h, device=device),       # (H,)
        torch.arange(w, device=device),       # (W,)
        indexing="ij"
    )
    # => y.shape == (H, W)
    # => x.shape == (H, W)

    if adaptive:
        sigma_x = max(1.0, bw / 6)
        sigma_y = max(1.0, bh / 6)
    else:
        sigma_x = sigma_y = 3.0

    # g: (H, W)
    g = torch.exp(-((x - cx)**2 / (2*sigma_x**2) +
                    (y - cy)**2 / (2*sigma_y**2)))
    return g


# ----------------------------------------------------------
# 2) 构建目标热图（多框，多类）
# ----------------------------------------------------------
def build_target_heatmap(gt_bboxes, gt_labels, class_names, h, w, device):
    B = len(gt_bboxes)         # batch size
    C = len(class_names)       # 类别数

    # target: (B, C, H, W)
    target = torch.zeros(B, C, h, w, device=device)

    # class_counts: (C,) 统计每个类别的 box 数量
    class_counts = torch.zeros(C, device=device)

    # 遍历 batch
    for i in range(B):
        # gt_bboxes[i] : (Ni, 4)
        # gt_labels[i] : (Ni,)
        # 遍历 每个 box
        for bbox, cls in zip(gt_bboxes[i], gt_labels[i]):
            x1, y1, x2, y2 = bbox    # 标量
            cx, cy = (x1 + x2)/2, (y1 + y2)/2
            bw, bh = (x2 - x1), (y2 - y1)

            # g: (H, W)
            g = gaussian_2d(h, w, cx, cy, bw, bh, adaptive=True, device=device)

            # target[i, cls]: (H, W)
            # 取最大值防止多个框重叠
            target[i, cls] = torch.maximum(target[i, cls], g)  # (H, W)

            class_counts[cls] += 1

    return target, class_counts   # target: (B, C, H, W), class_counts: (C,)


# ----------------------------------------------------------
# 3) Focal Loss + 类别均衡 + Hard Negative Mining + 多尺度一致性
# ----------------------------------------------------------
def advanced_gaussian_loss(pred_logits, target, class_counts,
                           alpha=0.25, gamma=2,
                           mining_ratio=3.0,
                           multi_scale=True):
    """
    pred_logits: (B, C, H, W) — 未经过 sigmoid 的 logits
    target:      (B, C, H, W) — 高斯监督热图
    class_counts: (C,) — 每类出现次数
    """
    B, C, H, W = pred_logits.shape

    # BCE: (B, C, H, W)
    bce = F.binary_cross_entropy_with_logits(pred_logits, target, reduction='none')

    # probs: (B, C, H, W)
    probs = torch.sigmoid(pred_logits)

    # pos_mask: (B, C, H, W) 是否是目标区域
    pos_mask = target > 0.1

    pt = torch.where(pos_mask, probs, 1 - probs)      # (B, C, H, W)
    alpha_t = torch.where(pos_mask, alpha, 1 - alpha) # (B, C, H, W)

    # focal_weight: (B, C, H, W)
    focal_weight = alpha_t * (1 - pt)**gamma

    # class_weights: (C,) → 每个类别一个权重
    class_weights = 1.0 / (class_counts + 1e-6)  # (C,)
    class_weights = class_weights / class_weights.mean()   # (C,)

    # 对每个类别通道加权
    # focal_weight[:, cls] shape: (B, H, W)
    for cls in range(C):
        focal_weight[:, cls] *= class_weights[cls]

    # focal_loss: (B, C, H, W)
    focal_loss = focal_weight * bce

    # ----------------------------
    # 正样本 loss
    # focal_loss[pos_mask] → 变成(Num_pos,) 一维
    # ----------------------------
    pos_loss = focal_loss[pos_mask].sum()

    # 负样本 loss: (Num_neg,)
    neg_loss = focal_loss[~pos_mask]

    num_pos = pos_mask.sum()

    # Hard Negative Mining
    if num_pos > 0:
        num_neg = min(int(num_pos * mining_ratio), neg_loss.numel())
        if num_neg > 0:
            # 取最难的 top-k
            # topk = num_neg
            # neg_loss.view(-1) → flatten
            hard_neg, _ = torch.topk(neg_loss.view(-1), num_neg)

            # 标量
            total_loss = pos_loss + hard_neg.sum()
            normalizer = num_pos + num_neg
            main_loss = total_loss / normalizer
        else:
            main_loss = pos_loss / num_pos
    else:
        # 没有正样本，则用平均
        main_loss = focal_loss.mean()

    # ----------------------------
    # 多尺度一致性
    # pred_down, target_down: (B, C, H/2, W/2)
    # ----------------------------
    if multi_scale and H >= 4 and W >= 4:
        pred_down = F.avg_pool2d(torch.sigmoid(pred_logits), 2)
        target_down = F.avg_pool2d(target, 2)

        consistency_loss = F.mse_loss(pred_down, target_down)

        # 标量
        main_loss = main_loss + 0.1 * consistency_loss

    return main_loss


# ----------------------------------------------------------
# ✅ 4) 可运行 DEMO
# ----------------------------------------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # pred_logits: (B, C, H, W)
    # B = 2, C = 3 类, H = W = 32
    pred_logits = torch.randn(2, 3, 32, 32, device=device, requires_grad=True)

    # gt_bboxes[i]: (Num_boxes, 4)
    # gt_labels[i]: (Num_boxes,)
    gt_bboxes = [
        torch.tensor([[5, 6, 15, 18], [10, 10, 20, 20]], device=device),
        torch.tensor([[3, 3, 12, 15], [15, 12, 28, 25]], device=device),
    ]
    gt_labels = [
        torch.tensor([0, 2], device=device),
        torch.tensor([1, 1], device=device)
    ]

    class_names = ["A", "B", "C"]

    # target: (B, C, H, W)
    # class_counts: (C,)     # # class_counts: (C,) 统计每个类别的 box 数量
    target, class_counts = build_target_heatmap(gt_bboxes, gt_labels, class_names, 32, 32, device)

    # loss: 标量 (1,)
    loss = advanced_gaussian_loss(pred_logits, target, class_counts)
    print("LOSS =", loss.item())

    # 反向传播
    loss.backward()
    print("Backward OK ✅")

    '''log
    LOSS = 0.8687697649002075
    Backward OK ✅
    '''
