import time
import torch
from torch import nn, optim

import sys
sys.path.append("..")

import utils as d2l

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Using {device} device and cuda is available:{torch.cuda.is_available()}")

class LeNet(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(LeNet, self).__init__(*args, **kwargs)
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=0), # in_channels, out_channels, kernel_size, padding
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2, stride=2), # kernel_size, stride
            nn.Conv2d(6, 16, kernel_size=5), # in_channels, out_channels, kernel_size
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2, stride=2) # kernel_size, stride
        )

        self.fc = nn.Sequential(
            nn.Linear(16*4*4, 120), # in_features, out_features
            nn.Sigmoid(),
            nn.Linear(120, 60),
            nn.Sigmoid(),
            nn.Linear(60, 10)
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output
    
net = LeNet()
print(net)


batch_size = 256
train_data, test_data = d2l.load_data_fashion_mnist_to_memory(batch_size=batch_size)

def evaluate_accuracy(data, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0

    with torch.no_grad():
        for X, y in data:
            if isinstance(net, torch.nn.Module):
                # 评估模式, 这会关闭dropout
                net.eval()
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                net.train() # 改回训练模式
            else: # 自定义的模型, 3.13节之后不会用到, 不考虑GPU
                if('is_training' in net.__code__.co_varnames): # 如果有is_training这个参数
                    # 将is_training设置成False
                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item() 
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item() 
            n += y.shape[0]
    return acc_sum / n


# 本函数已保存在d2lzh_pytorch包中方便以后使用
def train_ch5(net, train_data, test_data, batch_size, optimizer, device, num_epochs):
    net = net.to(device)
    print("training on ", device)
    loss = torch.nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        for X, y in train_data:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_data, net, device)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))


lr, num_epochs = 0.001, 5
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
train_ch5(net, train_data, test_data, batch_size, optimizer, device, num_epochs)


X,y = test_data[10]
true_labels = d2l.get_fashion_mnist_labels(y.numpy())
pred_labels = d2l.get_fashion_mnist_labels(net(X.to(device)).argmax(dim=1).cpu().numpy())
titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]

d2l.show_fashion_mnist(X[100:115], titles[100:115])