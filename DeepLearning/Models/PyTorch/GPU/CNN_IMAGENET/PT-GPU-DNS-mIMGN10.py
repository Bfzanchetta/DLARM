from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re
import torch
import torch.nn as nn
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
import argparse
import random
import shutil
import sys
import numpy as np
import torch.utils.model_zoo as model_zoo
import warnings
from collections import OrderedDict
warnings.filterwarnings("ignore")

plt.ion()
start = time.time()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
PATH = os.environ.get("DATASET_10SET", None)

epochs = 10

classes = ('n01440764', 'n01443537', 'n02037110', 'n02051845', 'n02125311', 'n02127052', 'n02799071', 'n03443371', 'n03929855', 'n04435653')

data_transform_train = transforms.Compose([
    transforms.RandomSizedCrop(299),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

miniImageNet_trainset = datasets.ImageFolder(root=PATH+'/train', transform=data_transform_train)

trainset_loader = torch.utils.data.DataLoader(miniImageNet_trainset, batch_size=4, shuffle=True, num_workers=4)

data_transform_val = transforms.Compose([
    transforms.Resize(331),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])

miniImageNet_validationset = datasets.ImageFolder(root=PATH+'/val', transform=data_transform_val)


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

##### INICIO MODELO #####



class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=10):

        super(DenseNet, self).__init__()

        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        self.classifier = nn.Linear(num_features, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        out = self.classifier(out)
        return out



##### FIM MODELO ######

loss_fc = torch.nn.CrossEntropyLoss().to(device)
densenet = DenseNet()
densenet = densenet.to(device)
optimizer = torch.optim.Adam(densenet.parameters(), lr=1e-4)


for i in range(epochs):
    since = time.time()
    densenet.train()
    running_loss = 0.0
    for j, (input, targets) in enumerate(trainset_loader):
        input, targets = Variable(input).to(device), Variable(targets).to(device)
        input = torch.autograd.Variable(input)
        train = densenet(input)
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
    densenet.eval()
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
            validate = densenet(images)
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
