import torch

A = torch.tensor([[1., 2.],
                  [3., 4.]])     # shape (2, 2)

B = torch.tensor([[0., 0.],
                  [1., 1.]])     # shape (2, 2)

D = torch.stack([A, B], dim=0)

print(D.shape)

print(D)
