# -*- coding: utf-8 -*-
# @Time    : 2021/8/25 20:28
# @Author  : Wu Weiran
# @File    : cgan_mnist.py
import os
import torch
import argparse
import torchvision.datasets
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image


class Discriminator(nn.Module):
    def __init__(self, conf):
        super(Discriminator, self).__init__()

        self.label_embedding = nn.Embedding(conf.n_classes, conf.n_classes)
        self.d_net = nn.Sequential(
            nn.Linear(conf.n_classes + conf.img_size ** 2, 512),
            nn.LeakyReLU(0.2),

            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2),

            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2),

            nn.Linear(512, 1)
        )

    def forward(self, image, label):
        d_in = torch.cat((image.view(image.size(0), -1), self.label_embedding(label)), dim=-1)
        d_out = self.d_net(d_in)
        return d_out


class Generator(nn.Module):
    def __init__(self, conf):
        super(Generator, self).__init__()

        self.label_embedding = nn.Embedding(conf.n_classes, conf.n_classes)
        self.g_net = nn.Sequential(
            nn.Linear(conf.n_classes + conf.latent_size, conf.hidden_size * 2),
            nn.LeakyReLU(0.2),

            nn.Linear(conf.hidden_size * 2, conf.hidden_size * 4),
            nn.BatchNorm1d(conf.hidden_size * 4, 0.8),
            nn.LeakyReLU(0.2),

            nn.Linear(conf.hidden_size * 4, conf.hidden_size * 8),
            nn.BatchNorm1d(conf.hidden_size * 8, 0.8),
            nn.LeakyReLU(0.2),

            nn.Linear(conf.hidden_size * 8, conf.hidden_size * 16),
            nn.BatchNorm1d(conf.hidden_size * 16, 0.8),
            nn.LeakyReLU(0.2),

            nn.Linear(conf.hidden_size * 16, conf.img_size ** 2),
            nn.Tanh()
        )

    def forward(self, noise, label, img_size):
        g_in = torch.cat((noise, self.label_embedding(label)), dim=-1)
        g_out = self.g_net(g_in)
        result = g_out.reshape(noise.size(0), 1, img_size, img_size)
        return result


def denorm(x):
    x = (x + 1) / 2
    return x


def train(conf):
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # transform
    transform = transforms.Compose([
        transforms.Resize((conf.img_size, conf.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Dataset
    train_data = torchvision.datasets.MNIST(
        root='../data/MINIST',
        train=True,
        transform=transform,
        download=True
    )

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

    # Loss and Optimizer
    criterion = nn.MSELoss().to(device)
    # 若使用BCELoss，必须判别器最后一层为sigmoid,否则会报错RuntimeError: CUDA error: device-side assert triggered
    d_optimizer = torch.optim.Adam(D.parameters(), lr=conf.learning_rate, betas=(0.5, 0.999))
    g_optimizer = torch.optim.Adam(G.parameters(), lr=conf.learning_rate, betas=(0.5, 0.999))

    total_step = len(dataloader)

    for epoch in range(conf.num_epoch):
        for i, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)

            z = torch.randn(images.size(0), conf.latent_size).to(device)  # random noise
            z_labels = torch.randint(conf.n_classes, (images.size(0),)).to(device)
            real_label = torch.ones(images.size(0), 1).to(device)
            fake_label = torch.zeros(images.size(0), 1).to(device)

            # Train the discriminator
            real_output = D(images, labels)
            d_real_loss = criterion(real_output, real_label)

            fake_images = G(z, z_labels, conf.img_size).detach()
            fake_output = D(fake_images, z_labels)
            d_fake_loss = criterion(fake_output, fake_label)

            d_loss = d_real_loss + d_fake_loss
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # Train the generator
            fake_images = G(z, z_labels, conf.img_size)
            fake_output = D(fake_images, z_labels)
            g_loss = criterion(fake_output, real_label)

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            if (i + 1) % 300 == 0:
                print(
                    "Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}".format(
                        epoch + 1, config.num_epoch, i + 1, total_step, d_loss.item(), g_loss.item(),
                        real_output.mean().item(), fake_output.mean().item()))

        fake_images = fake_images.reshape(-1, 1, conf.img_size, conf.img_size)
        save_image(denorm(fake_images), os.path.join(conf.sample_dir, 'fake_image{}.png'.format(epoch + 1)))

    torch.save(G, "Generator.pth")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--sample_dir', default='samples', help='sample directory path')
    parser.add_argument('--img_size', default=28, type=int, help='the size of image')
    parser.add_argument('--batch_size', default=64, type=int, help='the size of the batches')
    parser.add_argument('--n_classes', default=10, type=int, help='the classes of the dataset')
    parser.add_argument('--hidden_size', default=64, type=int, help='the size of hidden layer')
    parser.add_argument('--latent_size', default=100, type=int, help='the size of latent layer')
    parser.add_argument('--learning_rate', default=0.0002, type=int, help='the learning rate for optimizer')
    parser.add_argument('--num_epoch', default=200, type=int, help='number of epochs of training')

    config = parser.parse_args()

    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)

    train(config)
