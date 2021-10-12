import os
import random
import torch
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, root_dir, size):
        super(CustomDataset, self).__init__()

        self.root_dir = root_dir
        self.img_size = size
        self.lol_dir = os.path.join(self.root_dir, 'trainA')  # 训练图像中的低光图像路径
        self.norm_dir = os.path.join(self.root_dir, 'trainB')  # 训练图像中的正常光图像路径
        self.low_light = os.listdir(self.lol_dir)
        self.norm_light = os.listdir(self.norm_dir)

        self.lol_len = len(self.low_light)
        self.norm_len = len(self.norm_light)

    def __len__(self):
        return max(self.lol_len, self.norm_len)

    def __getitem__(self, idx):
        lol_name = self.low_light[idx % self.lol_len]
        norm_name = self.norm_light[idx % self.norm_len]

        lol_path = os.path.join(self.lol_dir, lol_name)
        norm_path = os.path.join(self.norm_dir, norm_name)

        lol_image = np.array(Image.open(lol_path).convert('RGB'))  # Image.open读取png格式图片时,会默认读入4通道(多出的是透明度通道)
        norm_image = np.array(Image.open(norm_path).convert('RGB'))

        transform = A.Compose(
            [
                A.RandomCrop(self.img_size, self.img_size),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ToTensorV2()
            ],
            additional_targets={"image0": "image"}
        )

        augmentations = transform(image=lol_image, image0=norm_image)
        lol_image = augmentations["image"]
        norm_image = augmentations["image0"]

        input_img = lol_image
        # 数据增强方式之一：调整图像亮度
        if random.random() < 0.5:
            times = random.randint(200, 400) / 100.
            input_img = (lol_image + 1) / 2. / times
            input_img = input_img * 2 - 1

        r, g, b = input_img[0] + 1, input_img[1] + 1, input_img[2] + 1  # 取值范围由[-1, 1]变为[0, 2]
        input_gray = 1. - (0.299 * r + 0.587 * g + 0.114 * b) / 2.  # 除以2使范围变为[0, 1], 再取补，从而获得注意力图
        input_gray = torch.unsqueeze(input_gray, 0)  # 扩充一个维度

        return {'A': input_img, 'B': norm_image, 'A_gray': input_gray, 'A_path': lol_path}


if __name__ == '__main__':
    root_dir = "../data/enlightengan"
    dataset = CustomDataset(root_dir, 320)

    data = dataset[0]
    print(data['A'].shape)
