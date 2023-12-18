import sys
import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms

sys.path.append("..")

from common_utils import inspect_tensors
from common_utils import DOWNLOAD_DATA_ROOT

# copied from https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/01-basics/pytorch_basics/main.py

# ================================================================== #
#                         Table of Contents                          #
# ================================================================== #

# 1. Basic autograd example 1               (Line 25 to 39)
# 2. Basic autograd example 2               (Line 46 to 83)
# 3. Loading data from numpy                (Line 90 to 97)
# 4. Input pipline                          (Line 104 to 129)
# 5. Input pipline for custom dataset       (Line 136 to 156)
# 6. Pretrained model                       (Line 163 to 176)
# 7. Save and load model                    (Line 183 to 189) 


# ================================================================== #
#                     1. Basic autograd example 1                    #
# ================================================================== #

# Create tensors.
x = torch.tensor(1.0, requires_grad=True)
w = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(3.0, requires_grad=True)

# Build a computational graph.
y = w * x + b

# Compute gradients.
y.backward()

# Print out the gradients.
print(f"x.grad = {x.grad}")
print(f"w.grad = {w.grad}")
print(f"b.grad = {b.grad}")
print(f"y.grad = {y.retain_grad()}")

inspect_tensors([x,w,b,y])


# ================================================================== #
#                    2. Basic autograd example 2                     #
# ================================================================== #

print('='*50)

# Create tensors of shape (10, 3) and (10, 2).
x = torch.randn(10, 3)
y = torch.randn(10, 2)
print("inspect x, y")
inspect_tensors([x,y])

# Build a fully connected layer.
linear = nn.Linear(3, 2)

print(f"linear.weight:{linear.weight}")
print(f"linear.bias:{linear.bias}")

# Build loss function and optimizer.
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(linear.parameters(), lr=0.01)

# Forward pass.
pred = linear(x)

# Compute loss.
loss = criterion(pred, y)
print(f"loss:{loss.item()}")

loss.backward()

# Print out the gradients.
print ('dL/dw: ', linear.weight.grad) 
print ('dL/db: ', linear.bias.grad)

# 1-step gradient descent.
optimizer.step()

# You can also perform gradient descent at the low level.
# linear.weight.data.sub_(0.01 * linear.weight.grad.data)
# linear.bias.data.sub_(0.01 * linear.bias.grad.data)

# Print out the loss after 1-step gradient descent.
pred = linear(x)
loss = criterion(pred, y)
print('loss after 1 step optimization: ', loss.item())


# ================================================================== #
#                     3. Loading data from numpy                     #
# ================================================================== #

print('='*50)

# Create a numpy array.
x = np.array([[1, 2], [3, 4]])

# Convert the numpy array to a torch tensor.
y = torch.from_numpy(x)

# Convert the torch tensor to a numpy array.
z = y.numpy()

print(f"type(x):{type(x)}, type(y):{type(y)},type(z):type(z)")
inspect_tensors([y])

# Create a numpy array.
x = np.array([[[1, 2, 2], [3, 4, 4],[5, 6.0, 6]],[[7, 8, 8],[9, 10, 10], [11, 12, 12]]])

# Convert the numpy array to a torch tensor.
y = torch.from_numpy(x)

# Convert the torch tensor to a numpy array.
z = y.numpy()

print(f"type(x):{type(x)}, type(y):{type(y)},type(z):type(z)")
inspect_tensors([y])

# ================================================================== #
#                         4. Input pipeline                           #
# ================================================================== #

# Download and construct CIFAR-10 dataset.
print('='*50)

train_dataset = torchvision.datasets.CIFAR100(root=DOWNLOAD_DATA_ROOT,
                                              train=True,
                                              transform=transforms.ToTensor(),
                                              download=True)

# Fetch one data pair (read data from disk).
image, label = train_dataset[0]
print (f"image.size():{image.size()}")
print (f"label:{label}")
inspect_tensors([image])

# Data loader (this provides queues and threads in a very simple way).
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                          batch_size=64,
                                          shuffle=True)
''' Uncomment this if you want to test it.

# When iteration starts, queue and thread start to load data from files.
data_iter = iter(train_loader)

# Mini-batch images and labels.
images, labels = next(data_iter)

print(f"type(images):{type(images)}, type(labels):{type(labels)}")

inspect_tensors([images, labels])

# Actual usage of the data loader is as below.
for images, labels in train_loader:
    # Training code should be written here.
    pass
'''