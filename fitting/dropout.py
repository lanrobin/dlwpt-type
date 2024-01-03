import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.append("..") 
import utils


def dropout(X, drop_prob):
    X = X.float()
    assert 0 <= drop_prob <= 1
    keep_prob = 1 - drop_prob

    if keep_prob == 0:
        return torch.zeros_like(X)
    mask = (torch.randn(X.shape) < keep_prob).float()

    return mask * X / keep_prob


X = torch.arange(16).view(2, 8)
dropX = dropout(X, 0)
print(f"X:{X}")
print(f"dropX:{dropX}")

dropX = dropout(X, 0.5)
print(f"X:{X}")
print(f"dropX:{dropX}")

dropX = dropout(X, 1.0)
print(f"X:{X}")
print(f"dropX:{dropX}")