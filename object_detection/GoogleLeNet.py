import torch
import torch.nn as nn
import torch.nn.functional as F


class Inception(nn.Module):
    """
    Inception module in GoogLeNet (with dimension reduction)
    """

    def __init__(self, in_channels,
                 ch1x1,
                 ch3x3_reduce, ch3x3,
                 ch5x5_reduce, ch5x5,
                 pool_proj):
        super(Inception, self).__init__()

        # branch 1: 1x1 conv
        self.branch1 = nn.Conv2d(in_channels, ch1x1, kernel_size=1)

        # branch 2: 1x1 reduce → 3x3
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3_reduce, kernel_size=1),
            nn.ReLU(True),
            nn.Conv2d(ch3x3_reduce, ch3x3, kernel_size=3, padding=1)
        )

        # branch 3: 1x1 reduce → 5x5
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, ch5x5_reduce, kernel_size=1),
            nn.ReLU(True),
            nn.Conv2d(ch5x5_reduce, ch5x5, kernel_size=5, padding=2)
        )

        # branch 4: pool → 1x1
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)

        # Depth concat: (N, C1+C2+C3+C4, H, W)
        return torch.cat([b1, b2, b3, b4], dim=1)


class GoogLeNet(nn.Module):
    def __init__(self):
        super(GoogLeNet, self).__init__()

        # stem
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(True),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )

        # Inception blocks
        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)

        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        # final classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, 1000)

    def forward(self, x):
        import ipdb; ipdb.set_trace()
        # x: (1, 3, 224, 224)
        x = self.conv1(x)          # -> (N, 64, H/4, W/4)
        x = self.conv2(x)          # -> (N, 192, H/8, W/8)

        x = self.inception3a(x)    # -> (N, 256, ...)
        x = self.inception3b(x)    # -> (N, 480, ...)
        x = self.maxpool(x)

        x = self.inception4a(x)    # -> (N, 512, ...)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool(x)

        x = self.inception5a(x)
        x = self.inception5b(x)

        x = self.avgpool(x)        # -> (N, 1024, 1, 1)
        x = torch.flatten(x, 1)    # -> (N, 1024)
        x = self.dropout(x)        # -> (N, 1024)
        x = self.fc(x)             # -> (N, 1000)

        return x

if __name__ == "__main__":
    net = GoogLeNet()
    x = torch.randn(1, 3, 224, 224)
    y = net(x)
    print("output shape:", y.shape)



