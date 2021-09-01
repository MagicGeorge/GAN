# -*- coding: utf-8 -*-
# @Time    : 2021/8/17 21:02
# @Author  : Wu Weiran
# @File    : dataset.py
import os
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset


class anime(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.path = os.path.join(self.root_dir, 'faces')
        self.img_path = os.listdir(self.path)

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.path, img_name)
        img = Image.open(img_item_path)

        # 将[0,255]的PIL格式转换为[0,1]的FloatTensor
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        img = transform(img)

        return img


if __name__ == '__main__':
    root_dir = "../data/AnimeDataset"
    dataset = anime(root_dir)
    pic = dataset[0]
    print(pic)
