import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import TransformerDecoder, TransformerDecoderLayer
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


# --------------------------------------------------
# 位置编码：简单可学习版
# --------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, h, w, d_model):
        super().__init__()
        self.row_embed = nn.Parameter(torch.randn(h, d_model // 2))  # (h, d/2)
        self.col_embed = nn.Parameter(torch.randn(w, d_model // 2))  # (w, d/2)

    def forward(self):
        import ipdb; ipdb.set_trace()
        H, W = self.row_embed.size(0), self.col_embed.size(0)   # (16, 16)
        pos = torch.cat([
            self.col_embed.unsqueeze(0).repeat(H, 1, 1),    # (w, d/2) -> (1, w, d/2) -> (h, w, d/2)
            self.row_embed.unsqueeze(1).repeat(1, W, 1),    # (h, d/2) -> (h, 1, d/2) (h, w, d/2)
        ], dim=-1)  # (h, w, d)

        return pos.reshape(H * W, -1)  # (HW, d)


# --------------------------------------------------
# Mini-DETR
# --------------------------------------------------
class MiniDETR(nn.Module):
    def __init__(self, num_queries=10, num_classes=5, d_model=128, h=16, w=16):
        super().__init__()
        self.h = h
        self.w = w
        self.d_model = d_model

        # 简化：输入特征直接映射到 d_model 维度
        self.input_proj = nn.Linear(1, d_model)

        # 位置编码
        self.pos_encoding = PositionalEncoding(h, w, d_model)

        # Encoder
        enc_layer = TransformerEncoderLayer(d_model, nhead=8)
        self.encoder = TransformerEncoder(enc_layer, num_layers=3)

        # Decoder
        dec_layer = TransformerDecoderLayer(d_model, nhead=8)
        self.decoder = TransformerDecoder(dec_layer, num_layers=3)


        # Object queries
        self.query_embed = nn.Parameter(torch.randn(num_queries, d_model))

        # 输出 FFN
        self.class_head = nn.Linear(d_model, num_classes + 1)   # 多 1 类：no-object
        self.bbox_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 4),
            nn.Sigmoid()  # 输出归一化 xywh
        )

    def forward(self, x):
        """
        x: (B, 1, H, W) 或任何能 reshape 到 (B, H*W, 1) 的矩阵
        """
        B = x.size(0)  

        # 展平
        x = x.reshape(B, self.h * self.w, 1)  # (B, HW, C)  # (1, 256, 1)
        x = self.input_proj(x)                # (B, HW, d)  # (1, 256, 128)

        # 加位置编码
        pos = self.pos_encoding()             # (HW, d)         # (256, 128)
        pos = pos.unsqueeze(1).repeat(1, B, 1)  # (HW, B, d)    # (256, 1, 128)
        x = x.permute(1, 0, 2) + pos            # (HW, B, d)    # (256, 1, 128)
        # for i in HW:
        #     for j in B:
        #         for k in d:
        #             x[i][j][k] += pos[i][j][k]

        # Encoder
        memory = self.encoder(x)              # (HW, B, d)

        # Decoder 输入：object queries
        # p self.query_embed.shape : (10, 128) 
        query = self.query_embed.unsqueeze(1).repeat(1, B, 1)  # (N, B, d)  # (10, 1, 128)
        # query:  (N, B, d)   (10, 1 128)
        # memory: (HW, B, d)  (256, 1, 128)
        hs = self.decoder(query, memory)        # (N, B, d)  # (10, 1, 128)

        # 输出
        classes = self.class_head(hs)           # (N, B, num_classes+1)  # (10, 1, 6)
        bboxes = self.bbox_head(hs)             # (N, B, 4)              # (10, 1, 4)

        return classes.permute(1, 0, 2), bboxes.permute(1, 0, 2)
        # 输出维度：(B, N, num_classes+1), (B, N, 4)


def compute_giou_matrix(pred_boxes, gt_boxes):
    """
    pred_boxes: (N, 4) in xywh normalized
    gt_boxes:   (M, 4) in xywh normalized

    return:
        giou_matrix: (N, M)
    """
    N = pred_boxes.size(0)
    M = gt_boxes.size(0)

    # (N, 4) -> (N, M, 4)
    pred = box_xywh_to_xyxy(pred_boxes).unsqueeze(1).repeat(1, M, 1)
    # (M, 4) -> (N, M, 4)
    target = box_xywh_to_xyxy(gt_boxes).unsqueeze(0).repeat(N, 1, 1)

    # Intersection
    x1 = torch.max(pred[..., 0], target[..., 0])
    y1 = torch.max(pred[..., 1], target[..., 1])
    x2 = torch.min(pred[..., 2], target[..., 2])
    y2 = torch.min(pred[..., 3], target[..., 3])

    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)

    # Areas
    area_pred = (pred[..., 2] - pred[..., 0]) * (pred[..., 3] - pred[..., 1])
    area_target = (target[..., 2] - target[..., 0]) * (target[..., 3] - target[..., 1])

    union = area_pred + area_target - inter
    iou = inter / (union + 1e-6)

    # Enclosing box C
    xc1 = torch.min(pred[..., 0], target[..., 0])
    yc1 = torch.min(pred[..., 1], target[..., 1])
    xc2 = torch.max(pred[..., 2], target[..., 2])
    yc2 = torch.max(pred[..., 3], target[..., 3])

    area_c = (xc2 - xc1) * (yc2 - yc1)
    giou = iou - (area_c - union) / (area_c + 1e-6)

    return giou   # shape: (N, M)



def box_xywh_to_xyxy(boxes):
    # boxes: (N, 4) with xywh normalized
    x, y, w, h = boxes.unbind(-1)
    x1 = x - 0.5 * w
    y1 = y - 0.5 * h
    x2 = x + 0.5 * w
    y2 = y + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


def giou_loss(pred, target):
    # pred/target: (N, 4) in xywh
    pred_xyxy = box_xywh_to_xyxy(pred)   # (N, 4)
    target_xyxy = box_xywh_to_xyxy(target)  # (N, 4)

    # Intersection
    x1 = torch.max(pred_xyxy[:, 0], target_xyxy[:, 0])
    y1 = torch.max(pred_xyxy[:, 1], target_xyxy[:, 1])
    x2 = torch.min(pred_xyxy[:, 2], target_xyxy[:, 2])
    y2 = torch.min(pred_xyxy[:, 3], target_xyxy[:, 3])

    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)

    # Union
    area_pred = (pred_xyxy[:, 2] - pred_xyxy[:, 0]) * (pred_xyxy[:, 3] - pred_xyxy[:, 1])
    area_target = (target_xyxy[:, 2] - target_xyxy[:, 0]) * (target_xyxy[:, 3] - target_xyxy[:, 1])
    union = area_pred + area_target - inter

    iou = inter / (union + 1e-6)

    # GIoU
    xc1 = torch.min(pred_xyxy[:, 0], target_xyxy[:, 0])
    yc1 = torch.min(pred_xyxy[:, 1], target_xyxy[:, 1])
    xc2 = torch.max(pred_xyxy[:, 2], target_xyxy[:, 2])
    yc2 = torch.max(pred_xyxy[:, 3], target_xyxy[:, 3])

    area_c = (xc2 - xc1) * (yc2 - yc1)
    giou = iou - (area_c - union) / (area_c + 1e-6)

    return 1 - giou    # (N,)


def hungarian_loss(pred_logits, pred_boxes, gt_boxes, gt_labels):
    """
    pred_logits: (N, C+1)
    pred_boxes:  (N, 4)   归一化 xywh
    gt_boxes:    (M, 4)   # (1, 4)
    gt_labels:   (M,)     # (1,)
    """
    N = pred_boxes.size(0)      # 10
    M = gt_boxes.size(0)        # 1

    # (N, M) 分类成本
    # pred_logits: (N, C+1)  # (10, 6)
    out_prob = pred_logits.softmax(-1)
    cls_cost = -out_prob[:, gt_labels]  # (N, 1)  # (10, 1)
    # cls_cost.shape : (N, 1)  # (10, 1)

    # (N, M) bbox L1
    box_cost = torch.cdist(pred_boxes, gt_boxes, p=1)  # (N, M)  # (10, 1)

    # 匈牙利cost
    giou_cost = 1 - compute_giou_matrix(pred_boxes, gt_boxes) 
    C = cls_cost + box_cost + giou_cost
    C = C.detach().cpu()  # (N, M)  # (10, 1)

    row_ind, col_ind = linear_sum_assignment(C)
    # row_ind.shape: (M,)  # (1,)
    # col_ind.shape: (M,)  # (1,)

    # 计算 loss
    # pred_logits[row_ind].shape: (M, C+1)  
    # gt_labels[col_ind].shape:  (M)  # (1,)
    # pred_boxes[row_ind].shape: (M, 4)  
    # gt_boxes[col_ind].shape:  (M, 4)
    loss_cls = F.cross_entropy(pred_logits[row_ind], gt_labels[col_ind])
    loss_box_l1 = F.l1_loss(pred_boxes[row_ind], gt_boxes[col_ind])
    loss_box_giou = giou_loss(pred_boxes[row_ind], gt_boxes[col_ind]).mean()
    loss_box = loss_box_l1 + 1.0 * loss_box_giou

    return loss_cls + loss_box



if __name__ == "__main__":
    B = 1
    X = torch.randn(B, 1, 16, 16)  # (B, C, H, W)  # (1, 1, 16, 16)

    gt_boxes = torch.tensor([[[0.5, 0.5, 0.4, 0.4]]])  # (B, num_gt, 4)  # (1, 1, 4)
    gt_labels = torch.tensor([[2]])  # 类别 2  # (B, num_gt)              # (1, 1)


    model = MiniDETR()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    for step in range(1000):
        classes, boxes = model(X)  # (B, N, C+1), (B, N, 4)  
        # classes: (1, 10, 6)  
        # boxes: (1, 10, 4)

        loss = hungarian_loss(
            classes[0], 
            boxes[0],
            gt_boxes[0],
            gt_labels[0]
        )

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 50 == 0:
            print(step, loss.item())


