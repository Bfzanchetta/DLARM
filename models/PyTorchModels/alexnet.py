'''AlexNet in Pytorch.'''
'''Breno Zanchetta'''

import torch
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self, numClasses):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4, bias=False)
        self.conv2 = nn.Conv2d(48, 128, kernel_size=5, stride=1, bias=False)
        self.conv3 = nn.Conv2d(128, 192, kernel_size=3, stride=1, bias=False)
        self.conv4 = nn.Conv2d(192, 192, kernel_size=3, stride=1, bias=False)
        self.conv5 = nn.Conv2d(192, 128, kernel_size=3, stride=1, bias=False)
        self.fc1   = nn.Linear(128*2*3*3,2048*2)
        self.fc2   = nn.Linear(2048*2, 2048*2)
        self.fc3   = nn.Linear(2048*2, numClasses)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = F.relu(self.conv5(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.dropout2d(out, 0.5)
        out = F.relu(self.fc2(out))
        out = F.dropout2d(out, 0.5)
        out = self.fc3(out)
        return out

#logic here
