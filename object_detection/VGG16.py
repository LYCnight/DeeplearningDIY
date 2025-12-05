import torch
import torch.nn as nn
import torch.nn.functional as F


class VGG16(nn.Module):
    def __init__(self, num_classes=1000):
        super(VGG16, self).__init__()

        # ------------- Feature Extractor（对应图中的卷积 + ReLU + 最大池化） -------------
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3,   64, kernel_size=3, padding=1), nn.ReLU(True),  
            nn.Conv2d(64,  64, kernel_size=3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # output: 112×112   # 

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(True),
            nn.Conv2d(128,128, kernel_size=3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 56×56

            # Block 3
            nn.Conv2d(128,256, kernel_size=3, padding=1), nn.ReLU(True),
            nn.Conv2d(256,256, kernel_size=3, padding=1), nn.ReLU(True),
            nn.Conv2d(256,256, kernel_size=3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 28×28

            # Block 4
            nn.Conv2d(256,512, kernel_size=3, padding=1), nn.ReLU(True),
            nn.Conv2d(512,512, kernel_size=3, padding=1), nn.ReLU(True),
            nn.Conv2d(512,512, kernel_size=3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 14×14

            # Block 5
            nn.Conv2d(512,512, kernel_size=3, padding=1), nn.ReLU(True),
            nn.Conv2d(512,512, kernel_size=3, padding=1), nn.ReLU(True),
            nn.Conv2d(512,512, kernel_size=3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 7×7
        )

        # ------------- Classifier（对应图中的 4096→4096→1000 全连接层） -------------
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),

            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),

            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        import ipdb; ipdb.set_trace()
        # x:  (1, 3, 224, 224)
        x = self.features(x)
        # x: (1, 512, 7, 7)
        x = x.view(x.size(0), -1)  # flatten  # (1, 25800)
        x = self.classifier(x)   # (1, 1000)
        return x


# ---------------- 运行 Demo ----------------
if __name__ == "__main__":
    model = VGG16(num_classes=1000)

    # 模拟输入 224×224 RGB 图片
    x = torch.randn(1, 3, 224, 224)

    out = model(x)

    print("输出 logits shape:", out.shape)
