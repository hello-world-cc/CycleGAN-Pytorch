import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
import random
import os


class MyDataset(Dataset):
    def __init__(self, transform,img_path,mode):
        super(MyDataset, self).__init__()
        self.transform = transform
        self.img_path = img_path
        self.mode = mode
        self.length = 0
        if self.mode == "train":
            self.A = "trainA"
            self.B = "trainB"
        else:
            self.A = "testA"
            self.B = "testB"
        self.img_name_list1 = os.listdir(os.path.join(self.img_path, self.A))
        random.shuffle(self.img_name_list1)
        self.img_name_list2 = os.listdir(os.path.join(self.img_path, self.B))
        random.shuffle(self.img_name_list2)

    def __getitem__(self, index):
        img1 = Image.open(os.path.join(self.img_path, self.A, self.img_name_list1[index]))
        img1 = self.transform(img1)
        img2 = Image.open(os.path.join(self.img_path, self.B, self.img_name_list2[index]))
        img2 = self.transform(img2)
        return img1, img2

    def __len__(self):
        if self.mode == "train":
            return 1066
        else:
            return 110

def get_loader(img_path,mode,batch_size,num_workers,crop_size,img_size):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.Resize((crop_size, crop_size)),
        torchvision.transforms.RandomCrop((img_size, img_size)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = MyDataset(transform,img_path,mode)
    return torch.utils.data.DataLoader(dataset=dataset, shuffle=True, batch_size=batch_size)


