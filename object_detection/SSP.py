import torch
import torch.nn as nn
import torch.nn.functional as F


class SPPLayer(nn.Module):
    def __init__(self, pool_sizes=[1, 2, 4]):
        super().__init__()
        self.pool_sizes = pool_sizes

    def forward(self, x):
        # x shape: (B, C, H, W)
        B, C, H, W = x.size()
        features = []

        for p in self.pool_sizes:
            # 自适应池化成 p×p（不管输入多大都能得到固定大小）
            pooled = F.adaptive_max_pool2d(x, output_size=(p, p))  # (B, C, p, p)
            features.append(pooled.view(B, -1))  # (b, c, p*p)

        return torch.cat(features, dim=1)



if __name__ == "__main__":
    spp = SPPLayer(pool_sizes=[1,2,4])

    x1 = torch.randn(1, 256, 20, 30)   # feature map A（大一点）
    x2 = torch.randn(1, 256, 13, 17)   # feature map B（小一点）

    y1 = spp(x1)
    y2 = spp(x2)

    print("y1 shape:", y1.shape)
    print("y2 shape:", y2.shape)
    '''
    y1 shape: torch.Size([1, 5376])
    y2 shape: torch.Size([1, 5376])
    '''


