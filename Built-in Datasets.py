# This method can be used for any built-in datasets in torchvision.datasets
# Check this link for list of built-in Datasets: https://pytorch.org/vision/stable/datasets.html

import torch
import torchvision
import torchvision.transforms as tt

from torch.utils.data import DataLoader

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# CIFAR10 dataset (images and labels)
train_dataset = torchvision.datasets.CIFAR10(root='./Dataset/CIFAR10', train=True, transform=tt.ToTensor(), download=True)

test_dataset = torchvision.datasets.CIFAR10(root='./Dataset/CIFAR10', train=False, transform=tt.ToTensor(), download=True)

# DataLoader (input pipeline)
batch_size = 100
train_dl = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=4, pin_memory=True)
test_dl = DataLoader(test_dataset, batch_size, num_workers=4, pin_memory=True)
