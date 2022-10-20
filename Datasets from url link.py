# This method can be used for datasets that can be directly downloaded from a link for ex: fast.ai
# Check this link for fast.ai Datasets: https://course.fast.ai/datasets

import os
import tarfile
import zipfile
import torchvision.transforms as tt

from torchvision.datasets.utils import download_url
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# Dowload the dataset
dataset_url = "https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz"
download_url(dataset_url, '.')

# Extract from archive (tgz file)
with tarfile.open('./cifar10.tgz', 'r:gz') as tar:
    def is_within_directory(directory, target):
        
        abs_directory = os.path.abspath(directory)
        abs_target = os.path.abspath(target)
    
        prefix = os.path.commonprefix([abs_directory, abs_target])
        
        return prefix == abs_directory
    
    def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
    
        for member in tar.getmembers():
            member_path = os.path.join(path, member.name)
            if not is_within_directory(path, member_path):
                raise Exception("Attempted Path Traversal in Tar File")
    
        tar.extractall(path, members, numeric_owner=numeric_owner) 
        
    
    safe_extract(tar, path="./data")

# Extract from archive (zip file)
# with zipfile.ZipFile("./xxxx.zip","r") as zip:
#     zip.extractall(path="./data")

# Dataset
root_dir = './data/cifar10'
classes = os.listdir(root_dir + "/train")
train_dataset = ImageFolder(root_dir+'/train', transform=tt.ToTensor())
test_dataset = ImageFolder(root_dir+'/test', transform=tt.ToTensor())

# DataLoader (input pipeline)
batch_size = 100
train_dl = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=4, pin_memory=True)
test_dl = DataLoader(test_dataset, batch_size, num_workers=4, pin_memory=True)
