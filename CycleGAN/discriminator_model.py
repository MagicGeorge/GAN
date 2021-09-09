# -*- coding: utf-8 -*-
# @Time    : 2021/9/9 15:09
# @Author  : Wu Weiran
# @File    : discriminator_model.py
import torch
from torch import nn


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(Block, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride, 1, padding_mode="reflect"),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, features=[64, 128, 256, 512]):
        super(Discriminator, self).__init__()

        # 输入维度是[N, 3, 256, 256]
        self.initial = nn.Sequential(
            nn.Conv2d(3, features[0], 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2)
        )

        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(Block(in_channels, feature, stride=1 if feature == features[-1] else 2))
            in_channels = feature
        # 输入维度是[N, 512, 31, 31]
        layers.append(nn.Conv2d(in_channels, 1, 4, 1, 1, padding_mode="reflect"))
        # 不可以直接self.model = layers, 这样layers中的parameters不会注册到网络中，当训练的时候不会更新
        self.model = nn.Sequential(*layers)
        self.out = nn.Sigmoid()

    def forward(self, x):
        x = self.initial(x)
        x = self.model(x)
        return self.out(x)


def test():
    img_channel = 3
    img_size = 256
    x = torch.randn((5, img_channel, img_size, img_size))
    model = Discriminator()
    pred = model(x)
    print(pred.shape)


if __name__ == '__main__':
    test()
