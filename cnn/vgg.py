import time
import torch
from torch import nn, optim

import sys
sys.path.append("..") 
import utils as d2l

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for i in range(num_convs):
        if i == 0:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)) # in_channels, out_channels, kernel_size, padding
        else:
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)) # in_channels, out_channels, kernel_size, padding

        layers.append(nn.ReLU())

    layers.append(nn.MaxPool2d(kernel_size=2, stride=2)) # kernel_size, stride
    return nn.Sequential(*layers)

conv_arch = ((1, 1, 64), (1, 64, 128), (2, 128, 256), (2, 256, 512), (2, 512, 512)) # (num_convs, in_channels, out_channels)
# 经过5个vgg_block, 宽高会减半5次, 变成 224/32 = 7
fc_features = 512 * 7 * 7 # c * w * h
fc_hidden_units = 4096 # 任意

def vgg(conv_arch, fc_features, fc_hidden_units=4096):
    net = nn.Sequential()
    # 卷积层部分
    for i, (num_conv, in_channels, out_channels) in enumerate(conv_arch):
        net.add_module("vgg_block_" + str(i+1), vgg_block(num_conv, in_channels, out_channels))

    # 全连接层部分
    net.add_module("fc", nn.Sequential(d2l.FlattenLayer(),
                                    nn.Linear(fc_features, fc_hidden_units),
                                    nn.ReLU(),
                                    nn.Dropout(0.5),
                                    nn.Linear(fc_hidden_units, fc_hidden_units),
                                    nn.ReLU(),
                                    nn.Dropout(0.5),
                                    nn.Linear(fc_hidden_units, 10)
    ))
    return net

net = vgg(conv_arch, fc_features, fc_hidden_units)

X = torch.rand(1, 1, 224, 224)

for name, blk in net.named_children():
    X = blk(X)
    print(f"{name} output shape:{X.shape}")


ratio = 4
small_conv_arch = [(1, 1, 64//ratio), (1, 64//ratio, 128//ratio), (2, 128//ratio, 256//ratio), 
                   (2, 256//ratio, 512//ratio), (2, 512//ratio, 512//ratio)]
net = vgg(small_conv_arch, fc_features // ratio, fc_hidden_units // ratio)
print(net)


batch_size = 128
# 如出现“out of memory”的报错信息，可减小batch_size或resize
train_data, test_data = d2l.load_data_fashion_mnist_to_memory(batch_size, resize=224)

lr, num_epochs = 0.001, 5
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
d2l.train_ch5(net, train_data, test_data, batch_size, optimizer, device, num_epochs)

origin_train_data, origin_test_data = d2l.load_data_fashion_mnist_to_memory(batch_size)

oX,y = origin_test_data[10]
X,_ = test_data[10]
true_labels = d2l.get_fashion_mnist_labels(y.numpy())
pred_labels = d2l.get_fashion_mnist_labels(net(X.to(device)).argmax(dim=1).cpu().numpy())
titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]

d2l.show_fashion_mnist(oX[100:115], titles[100:115])
