# -*- coding: utf-8 -*-
# @Time    : 2021/8/23 21:43
# @Author  : Wu Weiran
# @File    : wgan_gp.py
import os
import torch
import dataset
import argparse
from torch import nn
from torch import autograd
from torch import linalg
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

            nn.Conv2d(conf.hidden_size * 8, 1, 4, 1, 0)
        )

    def forward(self, x):
        x = self.d_net(x).view(-1)
        return x


class Generator(nn.Module):
    def __init__(self, conf):
        super(Generator, self).__init__()

        self.g_net = nn.Sequential(
            nn.ConvTranspose2d(conf.latent_size, conf.hidden_size * 8, 4, 1, 0),
            nn.ReLU(),

            nn.ConvTranspose2d(conf.hidden_size * 8, conf.hidden_size * 4, 4, 2, 1),
            nn.BatchNorm2d(conf.hidden_size * 4),
            nn.ReLU(),

            nn.ConvTranspose2d(conf.hidden_size * 4, conf.hidden_size * 2, 4, 2, 1),
            nn.BatchNorm2d(conf.hidden_size * 2),
            nn.ReLU(),

            nn.ConvTranspose2d(conf.hidden_size * 2, conf.hidden_size, 4, 2, 1),
            nn.BatchNorm2d(conf.hidden_size),
            nn.ReLU(),

            nn.ConvTranspose2d(conf.hidden_size, 3, 5, 3, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.g_net(x)
        return x


def denorm(x):
    x = (x + 1) / 2
    return x


def compute_gradient_penalty(interpolate, d_interpolate):
    gradients = autograd.grad(
        outputs=d_interpolate,
        inputs=interpolate,
        grad_outputs=torch.ones_like(d_interpolate),  # 输出为标量时，可以忽略该参数；输出为向量时，需要先将输出转为标量，分别求导
        # 例如：y对x求导
        #     x00  x01  x02
        # x = x10  x11  x12   ,    y = [y0, y1, y2]^T
        #     x20  x21  x22
        # 先把x看作[x0, x1, x2]^T, 相当于一个batch_size为3的batch
        #                d(y0/x0) d(y0/x1) d(y0/x2)
        # 雅可比矩阵记作J = d(y1/x0) d(y1/x1) d(y1/x2)  ,  其中不在主对角线上的元素一定为0
        #                d(y2/x0) d(y2/x1) d(y2/x2)
        # 右乘[1, 1, 1]^T, 就可以得到我们希望的结果[d(y0/x0), d(y1/x1), d(y2/x2)]^T
        # 右乘的这个矩阵就是这里的参数grad_outputs
        retain_graph=True,  # 保留计算图和中间变量梯度
        create_graph=True  # 在保留原图的基础上再建立额外的求导计算图，从而可以进行更高一阶的求导
    )[0]  # autograd.grad()得到的结果是一个元组, gradients.shape与interpolate.shape相同
    gradients = gradients.view(gradients.size(0), -1)
    gradients_penalty = ((linalg.norm(gradients, dim=1) - 1) ** 2).mean()
    return gradients_penalty


def train(conf):
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset
    train_data = dataset.anime(conf.dataset_dir)

    # DataLoader
    dataloader = DataLoader(
        dataset=train_data,
        batch_size=conf.batch_size,
        shuffle=True
    )

    # Discriminator
    D = Discriminator(conf).to(device)
    # Generator
    G = Generator(conf).to(device)

    # Optimizer
    d_optimizer = torch.optim.Adam(D.parameters(), lr=conf.learning_rate, betas=(0.5, 0.999))
    g_optimizer = torch.optim.Adam(G.parameters(), lr=conf.learning_rate, betas=(0.5, 0.999))

    total_step = len(dataloader)
    for epoch in range(conf.num_epoch):
        for i, images in enumerate(dataloader):
            images = images.to(device)

            # Train the discriminator
            real_output = D(images)
            d_real_loss = -torch.mean(real_output)

            z = torch.randn(images.size(0), conf.latent_size, 1, 1).to(device)
            fake_images = G(z).detach()
            fake_output = D(fake_images)
            d_fake_loss = torch.mean(fake_output)

            # torch.randn()返回的是x~N(0,1)的随机数，torch.rand()返回的是区间[0,1)内的随机数
            alpha = torch.rand(images.size(0), 1, 1, 1).to(device)
            # 沿着分别从P_r和P_g采样得到的点对之间的连线，随机插值采样
            interpolate = (alpha * images + ((1 - alpha) * fake_images)).requires_grad_(True)  # 保留梯度信息，保证梯度继续回传
            d_interpolate = D(interpolate)
            gradients_penalty = compute_gradient_penalty(interpolate, d_interpolate)

            d_loss = d_real_loss + d_fake_loss + conf.lambda_gp * gradients_penalty
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            if i % conf.n_critic == 0:
                # Train the generator
                z = torch.randn(images.size(0), conf.latent_size, 1, 1).to(device)
                gen_images = G(z)
                fake_output = D(gen_images)

                g_loss = -torch.mean(fake_output)
                g_optimizer.zero_grad()
                g_loss.backward()
                g_optimizer.step()

            if (i + 1) % 131 == 0:
                print(
                    'Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}'.format(
                        epoch + 1, conf.num_epoch, i + 1, total_step, d_loss.item(), g_loss.item(),
                        real_output.mean().item(), fake_output.mean().item()))
        save_image(denorm(gen_images), os.path.join(conf.sample_dir, 'fake_image{}.png'.format(epoch + 1)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_dir', default='../data/AnimeDataset/faces', help='dataset directory path')
    parser.add_argument('--sample_dir', default='samples', help='sample directory path')
    parser.add_argument('--batch_size', default=128, type=int, help='size of the batches')
    parser.add_argument('--hidden_size', default=64, type=int, help='the size of hidden layer')
    parser.add_argument('--latent_size', default=100, type=int, help='the size of input noise')
    parser.add_argument('--learning_rate', default=0.0001, type=float, help='the learning rate for optimizer')
    parser.add_argument('--num_epoch', default=1000, type=int, help='number of epochs of training')
    parser.add_argument('--lambda_gp', default=10, type=int, help='weight of gradient penalty')
    parser.add_argument('--n_critic', default=5, type=int, help='weight of gradient penalty')

    config = parser.parse_args()

    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)

    train(config)
