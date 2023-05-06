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
import warnings
warnings.filterwarnings("ignore")

plt.ion()
start = time.time()
device = torch.device("cpu")

epochs = 15

######

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

######

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


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
	x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
	x = self.fc2(x)
        return F.log_softmax(x, dim=1)

#def test():
#    net = LeNet()
#    x = torch.randn(2,3,32,32)
#	
#    y = net(x)
#    print(y.size())

#Loss Function Definition
#loss_fn = torch.nn.MSELoss(reduction='sum')
loss_fc = torch.nn.CrossEntropyLoss().to(device)
#criterion = nn.CrossEntropyLoss().cuda(args.gpu)
lenet = LeNet().to(device)
optimizer = torch.optim.SGD(lenet.parameters(), lr=1e-3, momentum=0.9)

for i in range(epochs):
        since = time.time()
        lenet.train()
        running_loss = 0.0
        for j, (input, targets) in enumerate(trainset_loader):
                input, targets = input.to(device), targets.to(device)
                input = torch.autograd.Variable(input)
		train = lenet(input)
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
        lenet.eval()
        running_loss = 0.0
        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        with torch.no_grad():
                end1 = time.time()
                for k, (images, labels) in enumerate(testset_loader):
                        images, labels = images.to(device), labels.to(device)
                        outputs = lenet(images)
                        loss = loss_fc(outputs, labels)
                        acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
                        losses.update(loss.item(), images.size(0))
                        top1.update(acc1[0], images.size(0))
                        top5.update(acc5[0], images.size(0))
                        batch_time.update(time.time() - end1)
                        end1 = time.time()
                        _, predicted = torch.max(outputs, 1)
                        c = (predicted == labels).squeeze()
                        for s in range(4):
                                label = labels[s]
                                class_correct[label] += c[s].item()
                                class_total[label] += 1
                        running_loss += loss.item()
                        if k % 16 == 15:
                                print('[%d, %5d] validation average loss: %.3f' %(i + 1, k + 1, running_loss / 16))
                                running_loss = 0.0
                                print('Test: [{0}/{1}]\t'
                                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(k, len(testset_loader), batch_time=batch_time, loss=losses, top1=top1, top5=top5))
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
        for k in range(10):
                print('Accuracy of %5s : %2d %%' % (classes[k], 100 * class_correct[k] / class_total[k]))
        time_elapsed1 = time.time() - end
        print('Validation complete in {:.0f}m {:.0f}s'.format(time_elapsed1 // 60, time_elapsed1 % 60))

totaltime = time.time() - start
print('Total execution complete in {:.0f}m {:.0f}s'.format(totaltime // 60, totaltime % 60))
configuration = totaltime - time_elapsed
configuration = configuration - time_elapsed1
print('Configuration complete in {:.0f}m {:.0f}s'.format(configuration // 60, configuration % 60))