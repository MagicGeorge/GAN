import time
import os

import torch

from options import TestOptions
from dataset import CustomDataset
from single_model import SingleModel
from torch.utils.data import DataLoader
import cv2 as cv

opt = TestOptions().parse()
opt.batch_size = 1  # test code only supports batchSize = 1

device = torch.device('cuda')
dataset = CustomDataset(opt.data_root, opt.fineSize)
dataloader = DataLoader(
        dataset=dataset,
        batch_size=opt.batch_size
)
model = SingleModel(opt, device)

# test
print(len(dataloader))
for i, data in enumerate(dataloader):
    model.set_input(data)
    visuals = model.predict()
    img_path = model.get_image_paths()
    print('process image... %s' % img_path)
    real = visuals['real_A']
    enhance = visuals['enhance_A']
    cv.imwrite('./real/real{}.jpg'.format(i), real)
    cv.imwrite('./result/enhance{}.jpg'.format(i), enhance)
