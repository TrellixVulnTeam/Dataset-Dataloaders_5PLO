# This method can be used for datasets that can be directly downloaded from a link for ex: fast.ai
# Check this link for fast.ai Datasets: https://course.fast.ai/datasets

import os
import tarfile
import torchvision.transforms as tt

from torchvision.datasets.utils import download_url
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# Dowload the dataset
dataset_url = "https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz"
download_url(dataset_url, '.')

# Extract from archive (tgz file)
with tarfile.open('./cifar10.tgz', 'r:gz') as tar:
    tar.extractall(path='./data')

# Dataset
root_dir = './data/cifar10'
classes = os.listdir(root_dir + "/train")
train_dataset = ImageFolder(root_dir+'/train', transform=tt.ToTensor())
test_dataset = ImageFolder(root_dir+'/test', transform=tt.ToTensor())

# DataLoader (input pipeline)
batch_size = 100
train_dl = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=4, pin_memory=True)
test_dl = DataLoader(test_dataset, batch_size, num_workers=4, pin_memory=True)
