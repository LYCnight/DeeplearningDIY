import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


def gaussian_2d(h, w, cx, cy, sigma_x, sigma_y):
    """生成二维高斯分布（热力图模板）"""
    # y: (H, 1), x: (1, W)
    # 最终 x, y 均为 (H, W)
    y, x = torch.meshgrid(
        torch.arange(h), torch.arange(w), indexing='ij'
    )

    print("y: ")
    print(y)
    '''
    [0, 0, 0, 0,
     1, 1, 1, 1,
     2, 2, 2, 2,
     3, 3, 3, 3,
     h, h, h, h,]
    '''


    print("x: ")
    print(x)
    '''
    [0, 1, 2, w,
     0, 1, 2, w,
     0, 1, 2, w,
     0, 1, 2, w,
     0, 1, 2, w,]
    '''

    # 按公式计算二维高斯分布
    g = torch.exp(-((x - cx)**2 / (2*sigma_x**2) +
                    (y - cy)**2 / (2*sigma_y**2)))
    # g: (H, W)
    return g


def gaussian_loss(pred_logits, gt_bbox, alpha=0.25, gamma=2.0):
    """
    pred_logits: (1, 1, H, W) —— 模型输出的 logits (未经过 sigmoid)   (B, C, H, W)
    gt_bbox: (x1, y1, x2, y2) —— GT 框坐标
    """
    _, _, H, W = pred_logits.shape  # 提取特征图尺寸

    # 解析 GT 框参数
    x1, y1, x2, y2 = gt_bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2     # (标量) 中心点坐标
    w, h = x2 - x1, y2 - y1                   # (标量) 宽高
    sigma_x, sigma_y = w / 6, h / 6           # 控制高斯扩散范围

    # === Step 1: 生成高斯真值热图 ===
    target = gaussian_2d(H, W, cx, cy, sigma_x, sigma_y)
    # target: (H, W)

    # === Step 2: Sigmoid 激活模型输出 ===
    pred = torch.sigmoid(pred_logits[0, 0])
    # pred_logits: (1, 1, H, W)
    # pred: (H, W) —— 概率分布（每个像素的“置信度”）

    # === Step 3: 计算 Binary Cross Entropy (逐点) ===
    bce = F.binary_cross_entropy(pred, target, reduction='none')
    # bce: (H, W) —— 每个像素的 BCE 损失

    # === Step 4: 计算 Focal Loss 权重 ===
    pt = torch.where(target > 0.1, pred, 1 - pred)
    # pt: (H, W)
    # 目标区域取 pred，背景区域取 1 - pred
    # → 表示该点预测正确的概率

    alpha_t = torch.where(target > 0.1, alpha, 1 - alpha)
    # alpha_t: (H, W)
    # 目标区域使用 alpha，背景使用 1 - alpha

    # === Step 5: Focal Loss 加权 ===
    loss = (alpha_t * (1 - pt)**gamma * bce).mean()
    # loss: 标量
    # (1 - pt)**γ 强调困难样本，γ=2 时效果最佳
    # mean() 表示对所有像素取平均损失

    return loss, target, pred


# ==== 测试 ====
if __name__ == '__main__':
    H, W = 64, 64
    pred_logits = torch.randn(1, 1, H, W)  # (1, 1, H, W)
    gt_bbox = torch.tensor([20, 25, 40, 45])  # (4,) —— x1, y1, x2, y2

    loss, target, pred = gaussian_loss(pred_logits, gt_bbox)

    print(f"Loss = {loss.item():.4f}")

    # ==== 可视化 ====
    plt.subplot(1, 2, 1)
    plt.title("Ground Truth Gaussian")
    plt.imshow(target, cmap='hot')
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.title("Model Prediction")
    plt.imshow(pred.detach(), cmap='hot')
    plt.colorbar()

    plt.show()
