from torchvision.datasets import CocoCaptions, CocoDetection, ImageFolder
from torchvision import transforms
from pycocotools.coco import COCO
import torchvision.models as torchmodels
import torch
import os
import json
from torchvision.transforms import ToPILImage
from IPython.display import Image
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
pylab.rcParams['figure.figsize'] = (8.0, 10.0)


class VggPretrainedNet(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        vgg16 = torchmodels.vgg16(pretrained=True)
        self.features = vgg16.features
        for p in self.features.parameters():
            p.requires_grad = False
        self.lin1 = nn.Linear(512*7*7, 200)
        self.lin2 = nn.Linear(200, 100)
        self.lin3 = nn.Linear(100, 8)

    def forward(self, x):
        x = self.features(x).view(-1, 512*7*7)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        return x


def train(net, data_loader, epochs=2):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(data_loader):
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')
    return net


# net = Net()
