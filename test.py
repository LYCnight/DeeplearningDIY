import torch
import torch.nn as nn
import pdb


if __name__ == "__main__":
    pdb.set_trace()
    a = torch.arange(9).view(3,3)
    # b = torch.arange(3)
    b = torch.arange(3).unsqueeze(1)   # (3,1)
    c = a + b





