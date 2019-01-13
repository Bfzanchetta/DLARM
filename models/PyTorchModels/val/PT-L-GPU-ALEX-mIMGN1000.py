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

#data_transform_val = transforms.Compose([
#        transforms.Resize(256),
#        transforms.CenterCrop(224),
#        transforms.ToTensor(),
#        transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                             std=[0.229, 0.224, 0.225])
#    ])

#Loading Dataset 
miniImageNet_trainset = datasets.ImageFolder(root='/home/nvidia/Desktop/Desenvolvimento/newset',
                                           transform=data_transform_train)

#miniImageNet_validationset = datasets.ImageFolder(root='/home/nvidia/miniset/val',
#                                           transform=data_transform_val)
                                           
trainset_loader = torch.utils.data.DataLoader(miniImageNet_trainset,
                                             batch_size=5, shuffle=True,
                                             num_workers=4)

#validationset_loader = torch.utils.data.DataLoader(miniImageNet_validationset,
#                                             batch_size=4, shuffle=True,
#                                             num_workers=4)

class AlexNet(nn.Module):
    
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

#def test():
#    net = AlexNet()
#    x = torch.randn(2,3,32,32)
#	
#    y = net(x)
#    print(y.size())

#Loss Function Definition
#loss_fn = torch.nn.MSELoss(reduction='sum')
loss_fc = torch.nn.CrossEntropyLoss()
#criterion = nn.CrossEntropyLoss().cuda(args.gpu)

alexnet = AlexNet().to(device)
optimizer = torch.optim.Adam(alexnet.parameters(), lr=1e-4)
alexnet.train()

for i in range(epochs):
        since = time.time()
        for j, (input, targets) in enumerate(trainset_loader):
                input, targets = input.to(device), targets.to(device)
                input = torch.autograd.Variable(input)
		train = alexnet(input)
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
