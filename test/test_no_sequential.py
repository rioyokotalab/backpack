import torch
from torch import nn

from backpack import extend, backpack
from backpack.extensions.secondorder import DiagGGN
from backpack.extensions import FAIL_SILENT


class LeNet(nn.Module):

    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.relu1 = nn.ReLU()
        self.max_pool2d1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.max_pool2d2 = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc3 = nn.Linear(16*5*5, 120)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc5 = nn.Linear(84, num_classes)

    def forward(self, x):
        h = self.relu1(self.conv1(x))
        h = self.max_pool2d1(h)
        h = self.relu2(self.conv2(h))
        h = self.max_pool2d2(h)
        h = self.flatten(h)
        h = self.relu3(self.fc3(h))
        h = self.relu4(self.fc4(h))
        out = self.fc5(h)

        return out


model = extend(LeNet())
loss_fn = extend(nn.CrossEntropyLoss())

bs = 10
x = torch.randn(bs, 3, 32, 32)
target = torch.randint(low=0, high=9, size=(bs,))

with backpack(DiagGGN(fail_mode=FAIL_SILENT)):
    loss = loss_fn(model(x), target)
    loss.backward()
