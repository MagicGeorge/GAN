# -*- coding: utf-8 -*-
# @Time    : 2021/8/22 16:53
# @Author  : Wu Weiran
# @File    : dataset.py
import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class anime(Dataset):
    def __init__(self, root):
        self.root_dir = root
        self.img_path = os.listdir(root)

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir, img_name)
        img = Image.open(img_item_path)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        img = transform(img)
        return img
