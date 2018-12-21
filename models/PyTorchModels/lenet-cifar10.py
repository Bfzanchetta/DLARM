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

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

trainset_loader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
#data_transform_val = transforms.Compose([
#        transforms.Resize(256),
#        transforms.CenterCrop(224),
#        transforms.ToTensor(),
#        transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                             std=[0.229, 0.224, 0.225])
#    ])

#Loading Dataset 

#miniImageNet_validationset = datasets.ImageFolder(root='/home/nvidia/miniset/val',
#                                           transform=data_transform_val)
                                           

#validationset_loader = torch.utils.data.DataLoader(miniImageNet_validationset,
#                                             batch_size=4, shuffle=True,
#                                             num_workers=4)

#Definition of the Model
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

#def test():
#    net = LeNet()
#    x = torch.randn(2,3,32,32)
#	
#    y = net(x)
#    print(y.size())

#Loss Function Definition
#loss_fn = torch.nn.MSELoss(reduction='sum')
loss_fc = torch.nn.CrossEntropyLoss()
#criterion = nn.CrossEntropyLoss().cuda(args.gpu)

lenet = LeNet().to(device)
optimizer = torch.optim.SGD(lenet.parameters(), lr=1e-3, momentum=0.9)
lenet.train()

for i in range(epochs):
        since = time.time()
        for j, (input, targets) in enumerate(trainset_loader):
                input, targets = input.to(device), targets.to(device)
                input = torch.autograd.Variable(input)
		train = lenet(input)
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
