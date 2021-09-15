# -*- coding: utf-8 -*-
# @Time    : 2021/9/10 15:56
# @Author  : Wu Weiran
# @File    : config.py
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_dir = "../data/horse2zebra/train"
val_dir = "../data/horse2zebra/val"
learning_rate = 1e-5
batch_size = 1
num_epoch = 10
num_workers = 4
lambda_cycle = 10
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_GEN_H = "genh.pth.tar"
CHECKPOINT_GEN_Z = "genz.pth.tar"
CHECKPOINT_DISC_H = "disch.pth.tar"
CHECKPOINT_DISC_Z = "discz.pth.tar"

transforms = A.Compose(
    [
        A.Resize(256, 256),
        A.HorizontalFlip(p=0.5),  # 水平翻转，p参数指定进行翻转的概率
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2()
    ],
    additional_targets={"image0": "image"}
)
