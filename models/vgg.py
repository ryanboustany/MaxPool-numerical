'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
import sys

from maxpool import *
from relu import *

cfg = {
    'VGG9': [64, 'M', 128, 'M', 256, 'M', 512, 'M', 512, 'M'],
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, maxpool_fn=nn.MaxPool2d, relu_fn=nn.ReLU, batch_norm=False, num_classes = 10):
        super(VGG, self).__init__()
        self.maxpool_fn = maxpool_fn
        self.relu_fn = relu_fn
        self.batch_norm = batch_norm
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, num_classes)
        
    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [self.maxpool_fn()]
            else:
                if self.batch_norm:
                    layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                            nn.BatchNorm2d(x),
                            self.relu_fn()]
                else:
                    layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                            self.relu_fn()]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
    
def test():
    net = VGG('VGG11',maxpool_fn=lambda: MaxPool2DBeta(0), relu_fn= lambda: ReLUAlpha(2))
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())
    print(net)
