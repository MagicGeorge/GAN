# -*- coding: utf-8 -*-
# @Time    : 2021/8/17 21:37
# @Author  : Wu Weiran
# @File    : GAN.py
import os
import argparse
import dataset
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image


class Discriminator(nn.Module):
    def __init__(self, conf):
        super(Discriminator, self).__init__()

        self.d_net = nn.Sequential(
            nn.Conv2d(3, conf.hidden_size, 5, 3, 1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(conf.hidden_size, conf.hidden_size * 2, 4, 2, 1),
            nn.BatchNorm2d(conf.hidden_size * 2),
            nn.LeakyReLU(0.2),

            nn.Conv2d(conf.hidden_size * 2, conf.hidden_size * 4, 4, 2, 1),
            nn.BatchNorm2d(conf.hidden_size * 4),
            nn.LeakyReLU(0.2),

            nn.Conv2d(conf.hidden_size * 4, conf.hidden_size * 8, 4, 2, 1),
            nn.BatchNorm2d(conf.hidden_size * 8),
            nn.LeakyReLU(0.2),

            nn.Conv2d(conf.hidden_size * 8, 1, 4, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.d_net(x).view(-1)  # view(-1)的作用是将[batch, C, H, W]的Tensor转变为[batch*C*H*W]
        return x


class Generator(nn.Module):
    def __init__(self, conf):
        super(Generator, self).__init__()

        self.g_net = nn.Sequential(
            # 输入是batch_size个latent_size * 1 * 1的噪声
            nn.ConvTranspose2d(conf.latent_size, conf.hidden_size * 8, 4, 1, 0),
            nn.BatchNorm2d(conf.hidden_size * 8),
            nn.ReLU(),
            # 输出是batch_size个(hidden_size*8) * 4 * 4的feature map
            # width_output = (w_in - 1)*s - 2p + k,
            # 这里的kernel_size、stride、padding等同于由输出尺寸经过Conv2d得到输入尺寸的对应参数值
            # 只是函数内部计算的时候会给输入的四周补上padding_new = kernel_size - 1 - padding， stride_new = 1/stride

            nn.ConvTranspose2d(conf.hidden_size * 8, conf.hidden_size * 4, 4, 2, 1),
            nn.BatchNorm2d(conf.hidden_size * 4),
            nn.ReLU(),
            # 输出是batch_size个(hidden_size*4) * 8 * 8的feature map

            nn.ConvTranspose2d(conf.hidden_size * 4, conf.hidden_size * 2, 4, 2, 1),
            nn.BatchNorm2d(conf.hidden_size * 2),
            nn.ReLU(),
            # 输出是batch_size个(hidden_size*2) * 16 * 16的feature map

            nn.ConvTranspose2d(conf.hidden_size * 2, conf.hidden_size, 4, 2, 1),
            nn.BatchNorm2d(conf.hidden_size),
            nn.ReLU(),
            # 输出是batch_size个hidden_size * 32 * 32的feature map

            nn.ConvTranspose2d(conf.hidden_size, 3, 5, 3, 1),
            nn.Tanh()  # 输出范围 -1~1
            # 输出形状：3 x 96 x 96
        )

    def forward(self, x):
        x = self.g_net(x)
        return x


def denorm(x):
    x = (x + 1) / 2
    return x


def train(config):
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset
    train_dataset = dataset.anime(config.dataset_dir)

    # DataLoader
    dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True
    )

    # Discriminator and Generator
    D = Discriminator(config).to(device)
    G = Generator(config).to(device)

    # Loss and Optimizer
    criterion = nn.BCELoss().to(device)
    d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
    g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))

    total_step = len(dataloader)
    # Train the model
    for epoch in range(config.num_epoch):
        for i, images in enumerate(dataloader):
            images = images.to(device)
            real_label = torch.ones(config.batch_size).to(device)
            fake_label = torch.zeros(config.batch_size).to(device)

            # Train Discriminator
            real_output = D(images)
            d_real_loss = criterion(real_output, real_label)

            z = torch.randn(config.batch_size, config.latent_size, 1, 1).to(device)
            fake_images = G(z).detach()  #
            fake_output = D(fake_images)
            d_fake_loss = criterion(fake_output, fake_label)
            d_loss = d_real_loss + d_fake_loss

            d_optimizer.zero_grad()
            g_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # Train Generator
            z = torch.randn(config.batch_size, config.latent_size, 1, 1).to(device)
            fake_images = G(z)
            fake_output = D(fake_images)
            g_loss = criterion(fake_output, real_label)

            d_optimizer.zero_grad()
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            if (i + 1) % 20 == 0:
                print(
                    "Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}".format(
                        epoch + 1, config.num_epoch, i + 1, total_step, d_loss.item(), g_loss.item(),
                        real_output.mean().item(), fake_output.mean().item()))

        fake_images = fake_images.reshape(-1, 3, 96, 96)
        save_image(denorm(fake_images), os.path.join(config.sample_dir, 'fake_image{}.png'.format(epoch + 1)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_dir', default='../data/AnimeDataset', help='dataset directory path')
    parser.add_argument('--sample_dir', default='samples', help='sample directory path')
    parser.add_argument('--batch_size', default=128, type=int, help='input batch size')
    parser.add_argument('--latent_size', default=100, type=int, help='the size of input noise')
    parser.add_argument('--hidden_size', default=64, type=int, help='the size of hidden layer')
    parser.add_argument('--num_epoch', default=5, type=int, help='number of epochs to train for')

    config = parser.parse_args()

    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)

    train(config)
