import torch
import torch.nn as nn
import sys
import os

from maxpool import *
from relu import *

class LeNet5(nn.Module):
    def __init__(self, num_classes=10, maxpool_fn=nn.MaxPool2d, relu_fn=nn.ReLU, batch_norm=False):
        super(LeNet5, self).__init__()
        self.batch_norm = batch_norm
        self.relu_fn = relu_fn
        self.maxpool_fn = maxpool_fn

        self.features = self._make_layers()
        self.classifier = nn.Sequential(
            nn.Linear(in_features=16*4*4, out_features=120),
            self.relu_fn(),
            nn.BatchNorm1d(120) if self.batch_norm else nn.Identity(),
            nn.Linear(in_features=120, out_features=84),
            self.relu_fn(),
            nn.BatchNorm1d(84) if self.batch_norm else nn.Identity(),
            nn.Linear(in_features=84, out_features=num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _make_layers(self):
        layers = [
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
            self.relu_fn(),
            self.maxpool_fn(),
            nn.BatchNorm2d(6) if self.batch_norm else nn.Identity(),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            self.relu_fn(),
            self.maxpool_fn(),
            nn.BatchNorm2d(16) if self.batch_norm else nn.Identity(),
        ]
        return nn.Sequential(*layers)


def test():
    net = LeNet5(maxpool_fn=lambda: MaxPool2DBeta(0))
    x = torch.randn(2,1,28,28)
    y = net(x)
    print(y.size())
    print(net)