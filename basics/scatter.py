import torch
import numpy as np


input = torch.tensor(np.arange(1, 11, 1.0), dtype=torch.float32).view(2, 5)
index=torch.tensor([[0, 1, 2, 0, 0], [2, 0, 0, 1, 2]])
output=torch.zeros(3, 5)
output.scatter_(0, index, input)
print(index)
print()
print(input)
print()
print(output)

output = torch.zeros(3, 5)
for i in range(2):
    output[index[i], i] = input[i]


input = torch.tensor(np.arange(1, 9, 1.0), dtype=torch.float32).view(2, 4)
output = torch.zeros(2, 5)
index = torch.tensor([[3, 2, 0, 1], [1, 2, 0, 3]])
output = output.scatter(1, index, input)
print("*"*30)
print(index)
print()
print(input)
print()
print(output)
