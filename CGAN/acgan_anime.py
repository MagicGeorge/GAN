# -*- coding: utf-8 -*-
# @Time    : 2021/8/26 20:27
# @Author  : Wu Weiran
# @File    : acgan_anime.py
import os
import csv
import argparse
import torch
from torch import nn
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image

# 头发颜色种类有12种， 眼睛颜色有11种
HAIRS = ['orange hair', 'white hair', 'aqua hair', 'gray hair', 'green hair', 'red hair', 'purple hair', 'pink hair',
         'blue hair', 'black hair', 'brown hair', 'blonde hair']
EYES = ['gray eyes', 'black eyes', 'orange eyes', 'pink eyes', 'yellow eyes', 'aqua eyes', 'purple eyes', 'green eyes',
        'brown eyes', 'red eyes', 'blue eyes']


def tag_preprocess(data_path):
    with open(os.path.join(data_path, 'tags_clean.csv'), 'r') as file:
        lines = csv.reader(file, delimiter=',')
        y_hairs = []
        y_eyes = []
        y_index = []
        for i, line in enumerate(lines):
            # idx对应的是图片的序号
            idx = line[0]
            # tags对应图片的所有特征，这里只关注eye和hair
            tags = line[1]

            tags = tags.split('\t')[:-1]  # 每行的最后都有多余的空格，通过切片去除
            y_hair = []  # 保存某一行中的hair特征
            y_eye = []  # 保存某一行中的eye特征
            for tag in tags:
                tag = tag[:tag.index(':')]
                if tag in HAIRS:
                    y_hair.append(HAIRS.index(tag))
                if tag in EYES:
                    y_eye.append(EYES.index(tag))
            # 如果同时存在hair和eye标签就代表这个标签是有用标签
            if len(y_hair) == 1 and len(y_eye) == 1:
                y_hairs.append(y_hair)
                y_eyes.append(y_eye)
                y_index.append(idx)

        return y_eyes, y_hairs, y_index


class anime(Dataset):
    def __init__(self, conf):
        self.root_path = conf.dataset_dir
        self.img_path = os.path.join(self.root_path, 'faces')
        self.eyes, self.hairs, self.index = tag_preprocess(conf.dataset_dir)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        img_name = self.index[idx]
        img_item_path = os.path.join(self.img_path, img_name + '.jpg')
        img = Image.open(img_item_path)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        img = transform(img)
        y_hair = torch.tensor(self.hairs[idx])
        y_eye = torch.tensor(self.eyes[idx])

        return img, y_hair, y_eye


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
            nn.LeakyReLU(),

            nn.Flatten()
        )
        # 输出真假概率
        self.label = nn.Sequential(
            nn.Linear(conf.hidden_size * 8 * 16, 1),
            nn.Sigmoid()
        )
        # 输出头发的颜色
        self.hair = nn.Linear(conf.hidden_size * 8 * 16, len(HAIRS))
        # 输出眼睛的颜色
        self.eye = nn.Linear(conf.hidden_size * 8 * 16, len(EYES))

    def forward(self, x):
        x = self.d_net(x)
        y_label = self.label(x)
        y_hair = self.hair(x)
        y_eye = self.eye(x)
        return y_label, y_hair, y_eye


class Generator(nn.Module):
    def __init__(self, conf):
        super(Generator, self).__init__()

        self.hair_embedding = nn.Embedding(len(HAIRS), conf.embedding_dim)
        self.eye_embedding = nn.Embedding(len(EYES), conf.embedding_dim)

        self.g_net = nn.Sequential(
            nn.ConvTranspose2d(conf.latent_size + 2 * conf.embedding_dim, conf.hidden_size * 8, 4, 1, 0),
            nn.BatchNorm2d(conf.hidden_size * 8),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(conf.hidden_size * 8, conf.hidden_size * 4, 4, 2, 1),
            nn.BatchNorm2d(conf.hidden_size * 4),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(conf.hidden_size * 4, conf.hidden_size * 2, 4, 2, 1),
            nn.BatchNorm2d(conf.hidden_size * 2),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(conf.hidden_size * 2, conf.hidden_size, 4, 2, 1),
            nn.BatchNorm2d(conf.hidden_size),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(conf.hidden_size, 3, 5, 3, 1),
            nn.Tanh()
        )

    def forward(self, x, y_hair, y_eye):
        hair_emb = self.hair_embedding(y_hair).reshape(y_hair.size(0), -1, 1, 1)
        eye_emb = self.eye_embedding(y_eye).reshape(y_eye.size(0), -1, 1, 1)
        x = torch.cat((hair_emb, x, eye_emb), dim=-3)
        x = self.g_net(x)
        return x


def denorm(x):
    x = (x + 1) / 2
    return x


def train(conf):
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset
    train_data = anime(conf)

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
    label_criterion = nn.BCELoss().to(device)
    feature_criterion = nn.CrossEntropyLoss().to(device)
    d_optimizer = torch.optim.Adam(D.parameters(), lr=conf.learning_rate, betas=(0.5, 0.999))
    g_optimizer = torch.optim.Adam(G.parameters(), lr=conf.learning_rate, betas=(0.5, 0.999))

    total_step = len(dataloader)
    for epoch in range(conf.num_epoch):
        for i, (images, hairs, eyes) in enumerate(dataloader):
            images = images.to(device)
            hairs = hairs.squeeze().to(device)  # 将维度[128, 1]降到一维[128]，这步操作是由CrossEntropyLoss决定的
            eyes = eyes.squeeze().to(device)

            real_label = torch.ones(images.size(0), 1).to(device)
            fake_label = torch.zeros(images.size(0), 1).to(device)

            # Train the Discriminator
            real_output, real_hairs, real_eyes = D(images)
            d_real_output_loss = label_criterion(real_output, real_label)
            d_real_hairs_loss = feature_criterion(real_hairs, hairs)
            d_real_eyes_loss = feature_criterion(real_eyes, eyes)
            d_real_loss = d_real_output_loss + d_real_hairs_loss + d_real_eyes_loss

            z = torch.randn(images.size(0), conf.latent_size, 1, 1).to(device)
            z_hairs = torch.randint(len(HAIRS), (images.size(0),)).to(device)
            z_eyes = torch.randint(len(EYES), (images.size(0),)).to(device)
            fake_images = G(z, z_hairs, z_eyes).detach()
            fake_output, fake_hairs, fake_eyes = D(fake_images)
            d_fake_output_loss = label_criterion(fake_output, fake_label)
            d_fake_hairs_loss = feature_criterion(fake_hairs, z_hairs)
            d_fake_eyes_loss = feature_criterion(fake_eyes, z_eyes)
            d_fake_loss = d_fake_output_loss + d_fake_hairs_loss + d_fake_eyes_loss
            d_loss = (d_real_loss + d_fake_loss) / 2

            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # Train the generator
            fake_images = G(z, z_hairs, z_eyes)
            fake_output, fake_hairs, fake_eyes = D(fake_images)
            if i + 2 == total_step:
                gen_images = fake_images

            g_fake_output_loss = label_criterion(fake_output, real_label)
            g_fake_hairs_loss = feature_criterion(fake_hairs, z_hairs)
            g_fake_eyes_loss = feature_criterion(fake_eyes, z_eyes)
            g_loss = g_fake_output_loss + g_fake_hairs_loss + g_fake_eyes_loss

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            if (i + 1) % 131 == 0:
                print(
                    'Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}'.format(
                        epoch + 1, conf.num_epoch, i + 1, total_step, d_loss.item(), g_loss.item(),
                        real_output.mean().item(), fake_output.mean().item()))
        save_image(denorm(gen_images), os.path.join(conf.sample_dir, 'fake_image{}.png'.format(epoch + 1)))

    torch.save(G, "anime_generator.pth")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_dir', default='../data/AnimeDataset/', help='dataset directory path')
    parser.add_argument('--sample_dir', default='samples', help='sample directory path')
    parser.add_argument('--batch_size', default=128, type=int, help='the size of the batches')
    parser.add_argument('--embedding_dim', default=8, type=int, help='the dim of embedding')
    parser.add_argument('--hidden_size', default=64, type=int, help='the size of hidden layer')
    parser.add_argument('--latent_size', default=100, type=int, help='the dimension of input noise')
    parser.add_argument('--img_size', default=96, type=int, help='the size of image')
    parser.add_argument('--dis_lr', default=0.0002, type=float, help='the learning rate for discriminator optimizer')
    parser.add_argument('--gen_lr', default=0.00015, type=float, help='the learning rate for generator optimizer')
    parser.add_argument('--num_epoch', default=10000, type=int, help='number of epochs of training')

    config = parser.parse_args()

    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)

    train(config)
