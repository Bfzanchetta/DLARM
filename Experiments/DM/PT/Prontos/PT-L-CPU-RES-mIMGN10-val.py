from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import math
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
import warnings
warnings.filterwarnings("ignore")

plt.ion()   
start = time.time()
device = torch.device("cpu")

epochs = 15

classes = ('n01440764', 'n02037110', 'n02125311', 'n02799071', 'n03443371', 'n03929855', 'n04435653', 'n01443537', 'n02051845', 'n02127052')

data_transform_train = transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

miniImageNet_trainset = datasets.ImageFolder(root='/home/nvidia/10set/train', transform=data_transform_train)

trainset_loader = torch.utils.data.DataLoader(miniImageNet_trainset, batch_size=4, shuffle=True, num_workers=4)

data_transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])

miniImageNet_validationset = datasets.ImageFolder(root='/home/nvidia/10set/val', transform=data_transform_val)


validationset_loader = torch.utils.data.DataLoader(miniImageNet_validationset, batch_size=4, shuffle=False, num_workers=4)

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


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


loss_fc = torch.nn.CrossEntropyLoss().to(device)
resnet = ResNet(Bottleneck, [3,8,36,3]).to(device)
optimizer = torch.optim.Adam(resnet.parameters(), lr=1e-4)


for i in range(epochs):
    since = time.time()
    resnet.train()
    running_loss = 0.0
    for j, (input, targets) in enumerate(trainset_loader):
        input, targets = input.to(device), targets.to(device)
        input = torch.autograd.Variable(input)
        train = resnet(input)
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
    resnet.eval()
    class_correct = list(0.0001 for r in range(10))
    class_total = list(0.0001 for t in range(10))
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    running_loss = 0.0
    with torch.no_grad():
        end1 = time.time()
        for k, (images, labels) in enumerate(validationset_loader):
            images, labels = images.to(device), labels.to(device)
            images = torch.autograd.Variable(images)
            #labels = torch.autograd.Variable(labels)
            validate = resnet(images)
            loss = loss_fc(validate, labels)
            acc1, acc5 = accuracy(validate, labels, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
            batch_time.update(time.time() - end1)
            end1 = time.time()
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
