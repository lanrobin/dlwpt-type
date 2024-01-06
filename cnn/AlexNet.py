import time
import torch
from torch import nn, optim
import torchvision

import sys
sys.path.append("..")
import utils as d2l

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AlexNet(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(AlexNet, self).__init__(*args, **kwargs)
        self.conv = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11, stride=4), # in_channels, out_channels, kernel_size, stride
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2), # kernel_size, stride
            nn.Conv2d(96, 256, kernel_size=5, stride = 1, padding=2), # in_channels, out_channels, kernel_size, stride, padding
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2), # kernel_size, stride
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1), # in_channels, out_channels, kernel_size, stride, padding
            nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1), # in_channels, out_channels, kernel_size, stride, padding
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1), # in_channels, out_channels, kernel_size, stride, padding
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2) # kernel_size, stride
        )
        self.fc = nn.Sequential(
            nn.Linear(256*5*5, 4096), # in_features, out_features
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 10)
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output
    
net = AlexNet()
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