from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from  torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms

import os
import argparse

#########
from torchvision import datasets,transforms, utils
from torch.utils.data import Dataset, DataLoader
from skimage import io, transform

import os
import torch
import time
import argparse
import random
import shutil
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
#import torch.nn.parallel
#import torch.backends.cudnn as cudnn
#import torch.distributed as dist
#import torch.optim
#import torch.multiprocessing as mp
#import torch.utils.data.distributed
#import torchvision.transforms as transforms
#import torchvision.datasets as datasets
#import torchvision.models as models
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

#device = torch.device("cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Parameters
epochs = 10

data_transform_train = transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

data_transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

#Loading Dataset 
miniImageNet_trainset = datasets.ImageFolder(root='/home/nvidia/miniset/train',
                                           transform=data_transform_train)

miniImageNet_validationset = datasets.ImageFolder(root='/home/nvidia/miniset/val',
                                           transform=data_transform_val)
                                           
trainset_loader = torch.utils.data.DataLoader(miniImageNet_trainset,
                                             batch_size=4, shuffle=True,
                                             num_workers=4)

validationset_loader = torch.utils.data.DataLoader(miniImageNet_validationset,
                                             batch_size=4, shuffle=True,
                                             num_workers=4)

#Definition of the Model

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(25088, 4096)
	self.classifier1 = nn.Linear(4096, 4096)
	self.classifier2 = nn.Linear(4096, 3)
	self.relu = nn.ReLU(True)
	self.dropout = nn.Dropout()	

    def forward(self, x):
        out = self.features(x)
	out = out.view(out.size(0), -1)
	out = self.classifier(out)
	out = self.relu(out)
	out = self.dropout(out)
	out = self.classifier1(out)
        out = self.relu(out)
        out = self.dropout(out)
	out = self.classifier2(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def test():
    net = VGG('VGG11')
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())

#Loss Function Definition
#loss_fn = torch.nn.MSELoss(reduction='sum')
loss_fc = torch.nn.CrossEntropyLoss()
#criterion = nn.CrossEntropyLoss().cuda(args.gpu)

vgg = VGG('VGG11').to(device)
optimizer = torch.optim.Adam(vgg.parameters(), lr=1e-4)
vgg.train()

for i in range(epochs):
        since = time.time()
        for j, (input, targets) in enumerate(trainset_loader):
                input, targets = input.to(device), targets.to(device)
                input = torch.autograd.Variable(input)
		train = vgg(input)
		print("Passo de Treinamento ",(i+1)*(j),"concluido.")
                targets = torch.autograd.Variable(targets)
		loss = loss_fc(train, targets)
		print("Loss:", loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
		print("Loss:", loss)
                
        print("So far so good")

time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

if __name__ == '__main__':
        main()
