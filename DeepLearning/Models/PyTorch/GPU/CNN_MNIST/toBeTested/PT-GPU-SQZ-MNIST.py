from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import os
import argparse
from torchvision import datasets,transforms, utils
from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
import time
import random
import shutil
import sys
import numpy as np
import torch.utils.model_zoo as model_zoo
import warnings
warnings.filterwarnings("ignore")

plt.ion()
start = time.time()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
PATH = os.environ.get("DATASET_10SET", None)

epochs = 10

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.1307,), (0.3081,))])

trainset = torchvision.datasets.MNIST(root='../data', train=True, download=True, transform=transform)
valset = torchvision.datasets.MNIST(root='../data', train=False, download=True, transform=transform)

trainset_loader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=4)
validationset_loader) = torch.utils.data.DataLoader(valset, batch_size=4, shuffle=False, num_workers=4)

classes = ('1', '2', '3', '4', '5', '6', '7', '8', '9', '0')

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

##### INICIO MODELO #####



class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)


class SqueezeNet(nn.Module):

    def __init__(self, version=1.0, num_classes=10):
        super(SqueezeNet, self).__init__()
        if version not in [1.0, 1.1]:
            raise ValueError("Unsupported SqueezeNet version {version}:"
                             "1.0 or 1.1 expected".format(version=version))
        self.num_classes = num_classes
        if version == 1.0:
            self.features = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=7, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(96, 16, 64, 64),
                Fire(128, 16, 64, 64),
                Fire(128, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 32, 128, 128),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(512, 64, 256, 256),
            )
        else:
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),
            )

        final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x.view(x.size(0), self.num_classes)



##### FIM MODELO ######

loss_fc = torch.nn.CrossEntropyLoss().to(device)
squeezenet = SqueezeNet()
squeezenet = squeezenet.to(device)
optimizer = torch.optim.Adam(squeezenet.parameters(), lr=1e-4)


for i in range(epochs):
    since = time.time()
    squeezenet.train()
    running_loss = 0.0
    for j, (input, targets) in enumerate(trainset_loader):
        input, targets = Variable(input).to(device), Variable(targets).to(device)
        input = torch.autograd.Variable(input)
        train = squeezenet(input)
        targets = torch.autograd.Variable(targets)
        loss = loss_fc(train, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if j % 100 == 99:
            print('[%d, %5d] training average loss: %.3f' %(i + 1, j + 1, running_loss / 100))
            running_loss = 0.0
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    end = time.time()
    squeezenet.eval()
    running_loss = 0.0
    class_correct = list(0.0001 for r in range(10))
    class_total = list(0.0001 for t in range(10))
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    with torch.no_grad():
        end1 = time.time()
        for k, (images, labels) in enumerate(validationset_loader):
            images, labels = images.to(device), labels.to(device)
            images = torch.autograd.Variable(images)
            validate = squeezenet(images)
            loss = loss_fc(validate, labels)
            acc1, acc5 = accuracy(validate, labels, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
            batch_time.update(time.time() - end1)
            end1 = time.time()
            labels = torch.autograd.Variable(labels)
            _, predicted = torch.max(validate, 1)
            c = (predicted == labels).squeeze()
            for m in range(labels.size(0)):
                label = labels[m]
                class_correct[label] += c[m].item()
                class_total[label] += 1
            running_loss += loss.item()
            if k % 16 == 15:
                print('[%d, %5d] validation average loss: %.3f' %(i + 1, k + 1, running_loss / 16))
                running_loss = 0.0
                print('Test: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(k, len(validationset_loader), batch_time=batch_time, loss=losses, top1=top1, top5=top5))
    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
    for k in range(10):
        print('Accuracy of %5s : %2d %%' % (classes[k], 100 * class_correct[k] / class_total[k]))

    validation_time = time.time() - end
    print('Validation complete in {:.0f}m {:.0f}s'.format(validation_time // 60, validation_time % 60))

totaltime = time.time() - start
print('Total execution complete in {:.0f}m {:.0f}s'.format(totaltime // 60, totaltime % 60))
configuration = totaltime - validation_time
configuration = configuration - time_elapsed
print('Configuration complete in {:.0f}m {:.0f}s'.format(configuration // 60, configuration % 60))
