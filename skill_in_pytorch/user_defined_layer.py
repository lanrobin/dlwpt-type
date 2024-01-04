import torch
from torch import nn

class CenteredLayer(nn.Module):
    def __init__(self, **kwargs):
        super(CenteredLayer, self).__init__(**kwargs)
    def forward(self, x):
        return x - x.mean()

layer = CenteredLayer()
y = layer(torch.tensor([1, 2, 3, 4, 5], dtype=torch.float))
print(f"y:{y}")

net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())
y = net(torch.rand(4, 8))
print(f"y: {y}, y.mean().item():{y.mean().item()}")