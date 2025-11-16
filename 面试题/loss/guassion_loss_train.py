import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


def gaussian_2d(h, w, cx, cy, sigma_x, sigma_y):
    """生成二维高斯分布"""
    y, x = torch.meshgrid(
        torch.arange(h), torch.arange(w), indexing='ij'
    )
    g = torch.exp(-((x - cx)**2 / (2*sigma_x**2) +
                    (y - cy)**2 / (2*sigma_y**2)))
    return g


def gaussian_loss(pred_logits, gt_bbox, alpha=0.25, gamma=2.0):
    """计算 Gaussian Focal Loss"""
    _, _, H, W = pred_logits.shape
    x1, y1, x2, y2 = gt_bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = x2 - x1, y2 - y1
    sigma_x, sigma_y = w / 6, h / 6

    target = gaussian_2d(H, W, cx, cy, sigma_x, sigma_y)
    pred = torch.sigmoid(pred_logits[0, 0])

    bce = F.binary_cross_entropy(pred, target, reduction='none')
    pt = torch.where(target > 0.1, pred, 1 - pred)
    alpha_t = torch.where(target > 0.1, alpha, 1 - alpha)
    loss = (alpha_t * (1 - pt)**gamma * bce).mean()
    return loss, target, pred


# ==== 模拟训练过程 ====
if __name__ == '__main__':
    H, W = 64, 64
    gt_bbox = torch.tensor([20, 25, 40, 45])
    loss_history = []

    # 随机初始化预测参数
    pred_logits = torch.randn(1, 1, H, W, requires_grad=True)
    optimizer = torch.optim.SGD([pred_logits], lr=0.5)

    for step in range(200):
        optimizer.zero_grad()
        loss, target, pred = gaussian_loss(pred_logits, gt_bbox)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())

        # 打印日志
        if (step + 1) % 20 == 0:
            print(f"Step [{step+1:03d}/200] Loss = {loss.item():.4f}")

        # 每隔 50 步可视化一次热图对比
        if (step + 1) % 50 == 0 or step == 0:
            plt.figure(figsize=(8, 3))
            plt.suptitle(f"Step {step+1} | Loss = {loss.item():.4f}")
            
            plt.subplot(1, 2, 1)
            plt.title("Ground Truth Gaussian")
            plt.imshow(target, cmap='hot')
            plt.axis("off")

            plt.subplot(1, 2, 2)
            plt.title("Model Prediction")
            plt.imshow(pred.detach(), cmap='hot')
            plt.axis("off")

            plt.tight_layout()
            plt.show()

    # ==== 绘制 Loss 曲线 ====
    plt.figure(figsize=(6,4))
    plt.plot(loss_history, label="Gaussian Loss", color='crimson')
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Gaussian Focal Loss over Training Steps")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()
