from torchvision import datasets,transforms, utils
from torch.utils.data import Dataset, DataLoader
import os
import torch
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

data_transform = transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

miniImageNet_dataset = datasets.ImageFolder(root='/home/nvidia/miniset/train',
                                           transform=data_transform)
                                           
dataset_loader = torch.utils.data.DataLoader(miniImageNet_dataset,
                                             batch_size=4, shuffle=True,
                                             num_workers=4)

print("It Works")
