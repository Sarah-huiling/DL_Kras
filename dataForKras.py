from __future__ import print_function
from torch.utils.data import DataLoader, Dataset, random_split

import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import cv2
import csv
import xlrd
from torchvision.transforms import transforms


def load_label_form_xl(xl_path):
    reads = xlrd.open_workbook(xl_path)
    label = []
    for row in range(reads.sheet_by_index(0).nrows):
        label.append(reads.sheet_by_index(0).cell(row, 0).value)  #从excel加载PID和labels
        label.append(reads.sheet_by_index(0).cell(row, 1).value)
    return label


def load_npy(x_path):
    label = [0, 0]
    data = np.load(x_path, allow_pickle=True)
    # labels = load_label_form_xl('/media/zhl/Local/ResearchData/Kras/labels.xlsx')
    labels = load_label_form_xl('/media/zhl/Local/ResearchData/Kras/labels.xlsx')

    #
    # if len(x_path.split('_')[-1]) > 8:
    #     path = x_path[:-8]
    # else:
    #     path = x_path[:-9]
    # id_index = labels.index(path.split('/')[-1])
    
    # path = x_path[:-4]  # PID
    # path = x_path[:-11]  #
    # id_index = labels.index(path.split('/')[-1])  # index
    id_index = labels.index(x_path)
    # one-hot
    if labels[id_index+1] == 1.0:
        label = [0, 1]
    if labels[id_index+1] == 0.0:
        label = [1, 0]
    return data, label

#print(load_npy('/media/stianci/360b04be-f526-4e91-aad3-85535b17af3e/3d_data/泪腺/resize_256/18-73755748_2anpy.npy'))

class MyDataset(Dataset):
    def __init__(self, root, transform=True):
        self.image_files = np.array(root)  
        self.transform = transform   

    def __getitem__(self, index):  # CPU tensor

        x, y = load_npy(self.image_files[index])  # 
        x = x.astype(np.int16)
        # CPU tensor
        return torch.FloatTensor(x), torch.FloatTensor(y)

    def __len__(self):
        return len(self.image_files)

    # 

    # def train_tf(x):
    #     im_aug = transforms.Compose([
    #         transforms.ToPILImage(),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor()
    #     ])
    #     x = im_aug(x)
    #     return x


# cv2.imshow('2',dataset.images[0,:,:,:])
# cv2.waitKey(0)
