# -*- coding: utf-8 -*-
# @Time    : 2021/9/9 23:41
# @Author  : Wu Weiran
# @File    : dataset.py
import os
import config
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class HorseZebraDataset(Dataset):
    def __init__(self, root_horse, root_zebra, transform=None):
        self.root_horse = root_horse
        self.root_zebra = root_zebra
        self.transform = transform

        self.horse_images = os.listdir(root_horse)
        self.zebra_images = os.listdir(root_zebra)

        self.horse_len = len(self.horse_images)
        self.zebra_len = len(self.zebra_images)
        self.length_dataset = max(self.horse_len, self.zebra_len)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, idx):
        horse_name = self.horse_images[idx % self.horse_len]
        zebra_name = self.zebra_images[idx % self.zebra_len]

        horse_path = os.path.join(self.root_horse, horse_name)
        zebra_path = os.path.join(self.root_zebra, zebra_name)

        horse_img = np.array(Image.open(horse_path).convert('RGB'))
        zebra_img = np.array(Image.open(zebra_path).convert('RGB'))

        if self.transform:
            augmentations = self.transform(image=horse_img, image0=zebra_img)
            horse_img = augmentations["image"]
            zebra_img = augmentations["image0"]

        return horse_img, zebra_img


if __name__ == '__main__':
    horse_path = "../data/horse2zebra/train/horse"
    zebra_path = "../data/horse2zebra/train/zebra"
    dataset = HorseZebraDataset(horse_path, zebra_path, config.transforms)

    horse, zebra = dataset[0]
    print(horse.shape)
    print(zebra.shape)
