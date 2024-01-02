import torch
from torch import nn
from torch.nn import init
import numpy as np
import time
import sys
sys.path.append("..") 
import utils

num_inputs, num_outputs, num_hiddens = 784, 10, 256

net = nn.Sequential(
        utils.FlattenLayer(),
        nn.Linear(num_inputs, num_hiddens),
        nn.ReLU(),
        nn.Linear(num_hiddens, num_outputs), 
        )

for params in net.parameters():
    init.normal_(params, mean=0, std=0.01)


batch_size = 256
train_iter, test_iter = utils.load_data_fashion_mnist(batch_size)

print("start to load data ...")
train_data = []
test_data = []
start = time.time()
for X, y in train_iter:
    train_data.append((X, y))

for X, y in test_iter:
    test_data.append((X, y))
print('%.2f sec to load the data.' % (time.time() - start))

loss = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(net.parameters(), lr=0.25)

num_epochs = 50
utils.train_ch3(net, train_data, test_data, loss, num_epochs, batch_size, None, None, optimizer)

print("starting testing ...")
X, y = next(iter(test_iter))
r_X = net(X[103:106])
print(r_X)
print(r_X.argmax(dim=1))
true_labels = utils.get_fashion_mnist_labels(y.numpy())
pred_labels = utils.get_fashion_mnist_labels(net(X).argmax(dim=1).numpy())
titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]

utils.show_fashion_mnist(X[100:115], titles[100:115])
print("finished testing ...")