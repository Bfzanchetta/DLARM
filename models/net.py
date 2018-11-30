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

device = torch.device("cpu")
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Parameters
#learning_rate = 1e-4
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

print("It Works")

#Definition of the Model
class LeNet(nn.Module):
    def __init__(self, num_classes=3):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

#Loss Function Definition
loss_fn = torch.nn.MSELoss(reduction='sum')
#criterion = nn.CrossEntropyLoss().cuda(args.gpu)

lenet = LeNet().to(device)
optimizer = torch.optim.Adam(lenet.parameters(), lr=1e-4)

for i in range(epochs):
        since = time.time()
        for i, (input, device) in enumerate(trainset_loader):
                train = lenet.forward(input)
                loss = loss_fn(train, device)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
        print("So far so good")

time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

if __name__ == '__main__':
        main()
