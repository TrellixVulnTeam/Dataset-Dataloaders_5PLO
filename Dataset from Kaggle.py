# This method can be used for any Kaggle datasets with its link
# Check this link for list of Kaggle Datasets: https://www.kaggle.com/datasets

# pip install opendatasets library
# pip install opendatasets --upgrade --quiet

import os
import shutil
import torch
import opendatasets as od
import torchvision.transforms as tt

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dowload the dataset
dataset_url = 'https://www.kaggle.com/alxmamaev/flowers-recognition'

if os.path.exists('./flowers-recognition'): # deletes a directory and all of its contents
    shutil.rmtree('./flowers-recognition')

od.download(dataset_url)

# Dataset
root_dir = './flowers-recognition/flowers'
os.listdir(root_dir) # lists all the folders in the directory
shutil.rmtree(root_dir + '/flowers') # to delete extra folder names 'flowers

transforms = tt.Compose([tt.Resize(64),
                         tt.RandomCrop(64), 
                         tt.ToTensor()])

train_dataset = ImageFolder(root_dir+'/train', transform=transforms)
test_dataset = ImageFolder(root_dir+'/test', transform=transforms)

# DataLoader (input pipeline)
batch_size = 100
train_dl = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=4, pin_memory=True)
test_dl = DataLoader(test_dataset, batch_size, num_workers=4, pin_memory=True)



