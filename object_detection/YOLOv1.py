import torch
import torch.nn as nn
import torch.nn.functional as F

# YOLO v1 默认参数
S = 7
B = 2
C = 20


class YOLOv1(nn.Module):
    def __init__(self):
        super().__init__()

        # 简化版 backbone
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),  # 224 -> 112
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),                        # 112 -> 56

            nn.Conv2d(64, 192, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),                        # 56 -> 28

            nn.Conv2d(192, 128, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 256, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),                        # 28 -> 14
        )

        # 继续简化
        self.conv_out = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),   # 👈 加这层！14×14 → 7×7
        )

        # 输出层：7×7×30
        self.fc = nn.Linear(1024 * S * S, S * S * (B * 5 + C))

    def forward(self, x):
        # x: (2, 3, 224, 224)   # (B, C, H, W)
        import ipdb; ipdb.set_trace()
        N = x.size(0)           
        x = self.features(x)         # (2, 512, 14, 14)
        x = self.conv_out(x)         # (2, 1024, 7, 7)
        x = x.view(N, -1)            # (2, 1024*7*7)
        x = self.fc(x)               # (2, 7*7*(2*5+20))
        return x.view(N, S, S, B * 5 + C)  # (2, 7, 7, 30)


class YOLOLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20, lambda_coord=5, lambda_noobj=0.5):
        super().__init__()
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def forward(self, preds, target):
        """
        preds:  [N, 7, 7, 30]
                每个 cell 输出：
                  B=2 个 box，每个 box 5 个数：x,y,w,h,conf  → 共 10
                  C=20 个类别概率                     → 共 20
                总共 10+20 = 30

        target: [N, 7, 7, 30]  同结构
        """

        # ---------------------------------------------------------
        # (1) 取出预测/GT（前 10 个是 bbox，后 20 个是类别 one-hot）
        # ---------------------------------------------------------

        # pred_boxes: [N, 7, 7, 2, 5]
        pred_boxes = preds[..., :10].reshape(-1, self.S, self.S, self.B, 5)

        # pred_cls:   [N, 7, 7, 20]
        pred_cls   = preds[..., 10:]

        # target_boxes: [N, 7, 7, 2, 5]
        target_boxes = target[..., :10].reshape(-1, self.S, self.S, self.B, 5)

        # target_cls:   [N, 7, 7, 20]
        target_cls   = target[..., 10:]


        # ---------------------------------------------------------
        # (2) obj_mask：哪些 bbox 是负责预测物体的？
        # target_boxes[..., 4] 是 GT 的 confidence
        # obj_mask: [N, 7, 7, 2]，bool 类型
        # ---------------------------------------------------------
        obj_mask = target_boxes[..., 4] > 0   # (N, 7, 7, 2)


        # ---------------------------------------------------------
        # (3) 坐标损失 (x,y,w,h)
        # pred_xy, pred_wh 形状都是 [N, 7, 7, 2, 2]
        # ---------------------------------------------------------

        pred_xy = pred_boxes[..., :2]
        target_xy = target_boxes[..., :2]

        pred_wh = pred_boxes[..., 2:4]
        target_wh = target_boxes[..., 2:4]

        # 只在 obj_mask 的 gridbox 上计算
        coord_loss = (
            F.mse_loss(pred_xy[obj_mask], target_xy[obj_mask]) +
            F.mse_loss(torch.sqrt(pred_wh[obj_mask].clamp(min=1e-6)),
                       torch.sqrt(target_wh[obj_mask]))
        )


        # ---------------------------------------------------------
        # (4) confidence 损失
        # pred_conf:   [N, 7, 7, 2]
        # target_conf: [N, 7, 7, 2]
        # ---------------------------------------------------------
        pred_conf   = pred_boxes[..., 4]
        target_conf = target_boxes[..., 4]

        # 有目标的 conf 损失
        obj_conf_loss = F.mse_loss(pred_conf[obj_mask], target_conf[obj_mask])

        # 无目标的 conf 损失（数量远大，需降低权重）
        noobj_conf_loss = F.mse_loss(pred_conf[~obj_mask], target_conf[~obj_mask])


        # ---------------------------------------------------------
        # (5) 分类损失
        # pred_cls:   [N, 7, 7, 20]
        # target_cls: [N, 7, 7, 20]
        #
        # 注意：分类损失只在 “含目标的 cell” 上计算！
        # obj_mask[..., 0] → [N, 7, 7]
        # ---------------------------------------------------------
        cls_loss = F.mse_loss(
            pred_cls[obj_mask[..., 0]],
            target_cls[obj_mask[..., 0]]
        )


        # ---------------------------------------------------------
        # (6) 总损失
        # ---------------------------------------------------------
        loss = (
            self.lambda_coord * coord_loss
            + obj_conf_loss
            + self.lambda_noobj * noobj_conf_loss
            + cls_loss
        )

        return loss



import torch
import numpy as np
import torchvision.ops as ops

def nms(boxes, scores, iou_thresh):
    return ops.nms(boxes, scores, iou_thresh)



def yolo_decode(pred, S=7, B=2, C=20, conf_thresh=0.2, nms_thresh=0.5):
    """
    pred: [1, 7, 7, 30]  YOLOv1 模型输出
    """

    pred = pred[0]  # 去掉 batch 维度 → [7,7,30]

    cell_size = 1.0 / S
    bboxes = []

    for i in range(S):
        for j in range(S):
            # 该 cell 全部数据：[30]
            cell = pred[i, j]

            # 取出两个 box： [2,5]
            boxes = cell[:10].reshape(B, 5)

            # 类别概率： [20]
            class_probs = cell[10:]

            class_id = torch.argmax(class_probs).item()
            class_prob = class_probs[class_id].item()

            for b in range(B):
                x, y, w, h, conf = boxes[b]

                # 最终得分
                score = conf.item() * class_prob
                if score < conf_thresh:
                    continue

                # 转为绝对坐标
                cx = (j + x.item()) * cell_size
                cy = (i + y.item()) * cell_size
                ww = w.item()
                hh = h.item()

                x1 = cx - ww / 2
                y1 = cy - hh / 2
                x2 = cx + ww / 2
                y2 = cy + hh / 2

                bboxes.append([x1, y1, x2, y2, score, class_id])

    if len(bboxes) == 0:
        return []

    # NMS 抑制冗余
    bboxes = torch.tensor(bboxes)
    keep = nms(bboxes[:, :4], bboxes[:, 4], nms_thresh)
    return bboxes[keep].tolist()



def test_model_forward():
    model = YOLOv1()
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print(y.shape)    # [2, 7, 7, 30]


def test_model_train():
    model = YOLOv1()
    criterion = YOLOLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for step in range(200):
        img = torch.randn(4, 3, 224, 224)
        target = torch.zeros(4, 7, 7, 30)

        # 假设某格子有目标
        target[0, 3, 4, 0:5] = torch.tensor([0.5, 0.5, 0.4, 0.3, 1])
        target[0, 3, 4, 10] = 1  # 类别 one-hot

        preds = model(img)
        loss = criterion(preds, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("step:", step, "loss:", loss.item())

def test_decode():
    model = YOLOv1()
    model.eval()

    img = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        pred = model(img)  # [1,7,7,30]

    boxes = yolo_decode(pred, S=7, B=2, C=20, conf_thresh=0.2)

    print("最终检测结果：")
    for box in boxes:
        x1, y1, x2, y2, score, cls = box
        print(f"class={cls}, score={score:.2f}, box=({x1:.2f},{y1:.2f},{x2:.2f},{y2:.2f})")


if __name__ == '__main__':
    # test_model_forward()
    test_model_train()

