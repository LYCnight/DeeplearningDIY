import torch
import torch.nn as nn
import torch.nn.functional as F

class RPN(nn.Module):
    def __init__(self, in_channels=512, mid_channels=256, num_anchors=9):
        super().__init__()

        # 3x3 conv (sliding window)
        self.conv = nn.Conv2d(in_channels, mid_channels, 3, padding=1)  # (1, 512, 30, 40) -> (1, 256, 30, 40)

        # classification: objectness (2k)
        self.cls_logits = nn.Conv2d(mid_channels, num_anchors * 2, 1)

        # regression: bbox deltas (4k)
        self.bbox_pred = nn.Conv2d(mid_channels, num_anchors * 4, 1)

        # init
        for l in [self.conv, self.cls_logits, self.bbox_pred]:
            nn.init.normal_(l.weight, std=0.01)
            nn.init.constant_(l.bias, 0)

    def forward(self, feature):
        # feature: [B, C, H, W]
        t = F.relu(self.conv(feature))

        # cls logits: [B, 2k, H, W]
        logits = self.cls_logits(t)

        # bbox regression: [B, 4k, H, W]
        bbox_reg = self.bbox_pred(t)

        return logits, bbox_reg


# test
if __name__ == "__main__":
    feature = torch.randn(1, 512, 30, 40)  # example FPN or conv feature map
    rpn = RPN(num_anchors=9)
    logits, bbox = rpn(feature)

    print("cls logits:", logits.shape)  # [1, 18, 30, 40]
    print("bbox pred:", bbox.shape)     # [1, 36, 30, 40]
