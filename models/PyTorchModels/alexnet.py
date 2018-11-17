#####################################
# AlexNet implementation in PyTorch #
# Adapted by Breno Zanchetta        #
#####################################
import torch.nn as nn
import torch.nn.functional as F

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 48, 11)
        self.conv2 = nn.Conv2d(48, 128, 5)
        self.conv3 = nn.Conv2d(128, 192, 3)
        self.conv4 = nn.Conv2d(192, 192, 3)
        self.conv5 = nn.Conv2d(192, 128, 3)
        #Parei aqui
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

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
