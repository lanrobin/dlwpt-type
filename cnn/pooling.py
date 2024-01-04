import torch
from torch import nn

def pool2d(X, pool_size, mode='max'):
    X = X.float()
    p_h, p_w = pool_size
    Y = torch.zeros(X.shape[0] - p_h + 1, X.shape[1] - p_w + 1)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i: i + p_h, j: j + p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i: i + p_h, j: j + p_w].mean()       
    return Y

X = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
Y = pool2d(X, (2, 2))

print(f"X:{X}\nY:{Y}")


Y = pool2d(X, (2, 2), mode='avg')

print(f"X:{X}\nY:{Y}")
X = torch.arange(16, dtype=torch.float).view((1, 1, 4, 4))

nn_pool2d = nn.MaxPool2d(3)
Y = nn_pool2d(X)

print(f"X:{X}\nY:{Y}")

X = torch.cat((X, X + 1), dim=1)
nn_pool2d = nn.MaxPool2d(3, padding=1, stride=2)
Y = nn_pool2d(X)

print(f"X:{X}\nY:{Y}")


nn_pool2d = nn.AvgPool2d(3, padding=1, stride=2)
Y = nn_pool2d(X)

print(f"X:{X}\nY:{Y}")