# -*- coding: utf-8 -*-
# @Time    : 2021/8/23 21:43
# @Author  : Wu Weiran
# @File    : dataset.py
import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class anime(Dataset):
    def __init__(self, dataset_dir):
        self.root_path = dataset_dir
        self.img_list = os.listdir(self.root_path)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        img_item_path = os.path.join(self.root_path, img_name)
        img = Image.open(img_item_path)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        img = transform(img)

        return img
