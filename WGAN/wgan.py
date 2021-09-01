# -*- coding: utf-8 -*-
# @Time    : 2021/8/22 15:36
# @Author  : Wu Weiran
# @File    : wgan.py
import os
import torch
import argparse
import dataset
import torchvision.datasets
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image


class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()

        self.d_net = nn.Sequential(
            # input [3, 96, 96]
            # 计算参数的时候先确定stride
            nn.Conv2d(3, config.hidden_size, 5, 3, 1),
            nn.LeakyReLU(0.2),
            # w_out = floor((w_in + 2*padding - k) / s + 1)
            # output [hidden_size, 32, 32]

            nn.Conv2d(config.hidden_size, config.hidden_size * 2, 4, 2, 1),
            nn.BatchNorm2d(config.hidden_size * 2),
            nn.LeakyReLU(0.2),
            # output [hidden_size*2, 16, 16]

            nn.Conv2d(config.hidden_size * 2, config.hidden_size * 4, 4, 2, 1),
            nn.BatchNorm2d(config.hidden_size * 4),
            nn.LeakyReLU(0.2),
            # output [hidden_size*4, 8, 8]

            nn.Conv2d(config.hidden_size * 4, config.hidden_size * 8, 4, 2, 1),
            nn.BatchNorm2d(config.hidden_size * 8),
            nn.LeakyReLU(0.2),
            # output [hidden_size*8, 4, 4]

            nn.Conv2d(config.hidden_size * 8, 1, 4, 1, 0)
            # 去掉sigmoid函数
        )

    def forward(self, x):
        x = self.d_net(x).view(-1)
        return x


class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()

        self.g_net = nn.Sequential(
            nn.ConvTranspose2d(config.latent_size, config.hidden_size * 8, 4, 1, 0),
            nn.BatchNorm2d(config.hidden_size * 8),
            nn.ReLU(),

            nn.ConvTranspose2d(config.hidden_size * 8, config.hidden_size * 4, 4, 2, 1),
            nn.BatchNorm2d(config.hidden_size * 4),
            nn.ReLU(),

            nn.ConvTranspose2d(config.hidden_size * 4, config.hidden_size * 2, 4, 2, 1),
            nn.BatchNorm2d(config.hidden_size * 2),
            nn.ReLU(),

            nn.ConvTranspose2d(config.hidden_size * 2, config.hidden_size, 4, 2, 1),
            nn.BatchNorm2d(config.hidden_size),
            nn.ReLU(),

            nn.ConvTranspose2d(config.hidden_size, 3, 5, 3, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.g_net(x)
        return x


def denorm(x):
    x = (x + 1) / 2
    return x


def train(conf):
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset
    train_dataset = dataset.anime(conf.dataset_dir)

    # DataLoader
    dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=conf.batch_size,
        shuffle=True,
    )

    # Discriminator and Generator
    D = Discriminator(conf).to(device)
    G = Generator(conf).to(device)

    # Optimizer
    d_optimizer = torch.optim.RMSprop(D.parameters(), lr=conf.learning_rate)
    g_optimizer = torch.optim.RMSprop(G.parameters(), lr=conf.learning_rate)

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

            # L = E_{x~P_r}(D(x)) - E_{z~P_z}(D(G(z))), L近似于真实分布与生成分布之间的Wasserstein距离
            d_loss = d_real_loss + d_fake_loss  # d_loss取的是L的反，优化器引导loss变小， 就相当于使L变大，提高判别器的区分能力
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # Clip weight of discriminator
            # 用于粗略地满足Lipschitz约束
            for p in D.parameters():
                p.data.clamp_(-conf.clip_value, conf.clip_value)

            if i % conf.n_critic == 0:
                # Train the generator
                z = torch.randn(images.size(0), conf.latent_size, 1, 1).to(device)
                fake_images = G(z)
                fake_output = D(fake_images)
                g_loss = -torch.mean(fake_output)  # g_loss就是定义为L，只是第一项与生成器无关，可有可无
                g_optimizer.zero_grad()
                g_loss.backward()
                g_optimizer.step()

                # 最后一个batch往往不是完整的，不方便可视化
                if i + 1 != total_step:
                    gen_images = fake_images

                if (i + 1) % 131 == 0:
                    print(
                        'Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}'.format(
                            epoch + 1, conf.num_epoch, i + 1, total_step, d_loss.item(), g_loss.item(),
                            real_output.mean().item(), fake_output.mean().item()))
        gen_images = gen_images.reshape(-1, 3, 96, 96)
        save_image(denorm(gen_images), os.path.join(conf.sample_dir, 'fake_image{}.png'.format(epoch + 1)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # type的常见类型有int, str, float, bool
    parser.add_argument('--dataset_dir', default='../data/AnimeDataset/faces', help='dataset directory path')
    parser.add_argument('--sample_dir', default='samples', help='sample directory path')
    parser.add_argument('--batch_size', default=128, type=int, help='size of the batches')
    parser.add_argument('--latent_size', default=100, type=int, help='the size of input noise')
    parser.add_argument('--hidden_size', default=64, type=int, help='the size of hidden layer')
    parser.add_argument('--learning_rate', default=0.0002, type=float, help='the learning rate for optimizer')
    parser.add_argument('--num_epoch', default=1000, type=int, help='number of epochs of training')
    parser.add_argument('--clip_value', default=0.01, help='lower and upper clip value for discriminator weights')
    parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")

    config = parser.parse_args()

    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)

    train(config)
