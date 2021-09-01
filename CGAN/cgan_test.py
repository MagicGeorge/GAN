# -*- coding: utf-8 -*-
# @Time    : 2021/8/26 19:44
# @Author  : Wu Weiran
# @File    : cgan_test.py
import os
import torch
import time
import numpy as np
from matplotlib import pyplot as plt
import cgan_mnist as cm
# from cgan_mnist import Generator
from acgan_anime import Generator

# 头发颜色种类有12种， 眼睛颜色有11种
HAIRS = ['orange hair', 'white hair', 'aqua hair', 'gray hair', 'green hair', 'red hair', 'purple hair', 'pink hair',
         'blue hair', 'black hair', 'brown hair', 'blonde hair']
EYES = ['gray eyes', 'black eyes', 'orange eyes', 'pink eyes', 'yellow eyes', 'aqua eyes', 'purple eyes', 'green eyes',
        'brown eyes', 'red eyes', 'blue eyes']


# def cgan():
#     # Device configuration
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#     # Load the model
#     G = torch.load("mnist_generator.pth")
#
#     # initialize the noise and label
#     labels = torch.randint(10, (10,)).to(device)
#     noise = torch.randn(10, 100).to(device)
#
#     start = time.time()
#     gen_image = G(noise, labels, 28)
#     gen_image = gen_image.reshape(-1, 1, 28, 28)
#
#     end_time = (time.time() - start)
#
#     r, c = 2, 5
#     fig, axs = plt.subplots(r, c)
#     cnt = 0
#     for i in range(r):
#         for j in range(c):
#             img = np.transpose(cm.denorm(gen_image).cpu().numpy(), (0, 2, 3, 1))
#             axs[i, j].imshow(img[cnt, :, :, 0], cmap='gray')
#             axs[i, j].set_title("Digit: %d" % labels[cnt])
#             # axs[i, j].axis('off')
#             cnt += 1
#     fig.savefig("images.png")
#
#     return end_time

def cgan():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the model
    G = torch.load("anime_generator.pth")

    # initialize the noise and label
    hairs = torch.randint(len(HAIRS), (10,)).to(device)
    eyes = torch.randint(len(EYES), (10,)).to(device)
    noise = torch.randn(10, 100, 1, 1).to(device)

    start = time.time()
    gen_image = G(noise, hairs, eyes)
    gen_image = gen_image.reshape(-1, 3, 96, 96)

    end_time = (time.time() - start)

    r, c = 2, 5
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            img = np.transpose(cm.denorm(gen_image).cpu().numpy(), (0, 2, 3, 1))
            axs[i, j].imshow(img[cnt, :, :, 0], cmap='gray')
            axs[i, j].set_title("Hair: {}, Eye: {}".format(HAIRS[hairs[cnt]], EYES[eyes[cnt]]))
            # axs[i, j].axis('off')  # 关闭坐标轴，即隐藏刻度和值
            cnt += 1
    fig.savefig("images.png")

    return end_time


if __name__ == '__main__':
    with torch.no_grad():
        sum_time = 0
        sum_time = sum_time + cgan()
        print('Time cost: {:.4f}'.format(sum_time))
