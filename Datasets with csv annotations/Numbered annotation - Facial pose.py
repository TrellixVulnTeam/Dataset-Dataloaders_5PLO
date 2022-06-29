# This method can be used for datasets that have seperate image folders and csv annotations.

import os
import torch
from skimage import io
import zipfile
import pandas as pd
import numpy as np
import torchvision.transforms as tt

from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset, DataLoader, random_split

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dowload the dataset
dataset_url = "url link to your dataset"
download_url(dataset_url, '.')

# Extract from archive (zip file)
with zipfile.ZipFile("./xxxx.zip","r") as zip:
    zip.extractall(path="./data")

# Dataset Class
class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
        image = io.imread(img_name)
        labels = torch.tensor(int(self.annotations.iloc[idx, 1]))

        if self.transform:
            image = self.transform(image)

        return (image, labels)

# Transformations
transforms = tt.Compose([tt.ToPILImage(),
                         tt.Resize(224),
                         tt.RandomCrop(size=224, padding=4, padding_mode="reflect"),
                         tt.ToTensor()])

# Load Data
dataset = CustomDataset(csv_file="csv file location",
                        root_dir="image folder location",
                        transform=transforms)

train_dataset, test_dataset = random_split(dataset, [50000, 10000])

# DataLoader (input pipeline)
batch_size = 100
train_dl = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=4, pin_memory=True)
test_dl = DataLoader(test_dataset, batch_size, num_workers=4, pin_memory=True)