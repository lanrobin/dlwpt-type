import torch
import numpy as np
import sys
sys.path.append("..")
import utils
import time

batch_size = 256
train_iter, test_iter = utils.load_data_fashion_mnist(batch_size)

num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_hiddens)), dtype=torch.float, requires_grad=True)
b1 = torch.zeros(num_hiddens, dtype=torch.float, requires_grad=True)
W2 = torch.tensor(np.random.normal(0, 0.01, (num_hiddens, num_outputs)), dtype=torch.float, requires_grad=True)
b2 = torch.zeros(num_outputs, dtype=torch.float, requires_grad=True)

def relu(X):
    return torch.max(input=X, other=torch.tensor(0.0))

def net(X):
    X = X.view((-1, num_inputs))
    H = relu(torch.matmul(X, W1) + b1)
    return torch.matmul(H, W2) + b2

loss = torch.nn.CrossEntropyLoss()

num_epochs, lr = 10, 100.0

print("start to load data ...")
train_data = []
test_data = []
start = time.time()
for X, y in train_iter:
    train_data.append((X, y))

for X, y in test_iter:
    test_data.append((X, y))
print('%.2f sec to load the data.' % (time.time() - start))

utils.train_ch3(net, train_data, test_data, loss, num_epochs, batch_size, [W1, b1, W2, b2], lr)

print("starting testing ...")
X, y = next(iter(test_iter))

true_labels = utils.get_fashion_mnist_labels(y.numpy())
pred_labels = utils.get_fashion_mnist_labels(net(X).argmax(dim=1).numpy())
titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]

utils.show_fashion_mnist(X[0:9], titles[0:9])
print("finished testing ...")