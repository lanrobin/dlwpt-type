import torch
import numpy as np
import random

num_inputs = 2
num_examples = 1000

true_w = [2, -3.4]
true_b = 4.2

features = torch.randn(num_examples, num_inputs, dtype=torch.float32)

labels = true_w[0] * features[:, 0] + true_w[1]*features[:, 1] +  true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float32)

print(features[0], labels[0])

def data_iter(batch_size, features, labels):
    nums = len(features)
    indices = list(range(nums))
    random.shuffle(indices)
    for i in range(0, nums, batch_size):
        j = torch.LongTensor(indices[i:min(i+batch_size, nums)])
        yield features.index_select(0, j), labels.index_select(0, j)

batch_size = 2
'''for X, y in data_iter(batch_size, features, labels):
    print(X, y)
    break
    '''
w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)), dtype=torch.float32)
b = torch.zeros(1, dtype=torch.float32)

print("*" * 50, "init w, b", "*" * 50)
print(w, b)
print("*" * 50, "init w, b", "*" * 50)

w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)

def linreg(X, w, b):
    return torch.mm(X, w) + b

def squared_loss(y_hat, y):
    return (y_hat - y.view(y_hat.size())) **2/2

def sgd(params, lr, batch_size):
    for param in params:
        param.data -= lr * param.grad / batch_size


# training
        
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss
step = sgd

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        y_hat = net(X, w, b)
        l = loss(y_hat, y)
        ls = l.sum()
        ls.backward()
        step([w, b], lr, batch_size)

        w.grad.data.zero_()
        b.grad.data.zero_()

    train_l = loss(net(features, w, b), labels)
    print( "*" * 100)
    print(f'epoch {epoch + 1}, loss {train_l.mean().item()}')

    print(true_w, "\n", w)
    print(true_b, "\n", b)