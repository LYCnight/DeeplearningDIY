import torch

A = torch.tensor([[1., 2.],
                  [3., 4.]])     # shape (2, 2)

B = torch.tensor([[0., 0.],
                  [1., 1.],
                  [2., 2.]])     # shape (3, 2)

D = torch.cdist(A, B, p=2)   # 欧氏距离(Euclidean)

print(D)
