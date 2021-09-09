# -*- coding: utf-8 -*-
# @Time    : 2021/9/1 19:33
# @Author  : Wu Weiran
# @File    : generator_model.py
import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_act=True, **kwargs):
        super(ConvBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, padding_mode="reflect", **kwargs)
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True) if use_act else nn.Identity()  # nn.Identity()不进行任何操作，将输入原样输出
        )

    def forward(self, x):
        return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            ConvBlock(channels, channels, kernel_size=3, stride=1, padding=1),
            ConvBlock(channels, channels, use_act=False, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(self, img_channels, num_feature=64, num_residual=9):
        super(Generator, self).__init__()

        self.initial = nn.Sequential(
            nn.Conv2d(img_channels, num_feature, 7, 1, 3, padding_mode="reflect"),
            nn.ReLU(inplace=True)
        )

        self.down_blocks = nn.ModuleList([  # nn.ModuleList是Python自带的list的升级版，加入到ModuleList里面的module是会注册到整个网络上的
            ConvBlock(num_feature, num_feature * 2, kernel_size=3, stride=2, padding=1),
            ConvBlock(num_feature * 2, num_feature * 4, kernel_size=3, stride=2, padding=1)
        ])

        self.residual_block = nn.Sequential(
            *[ResidualBlock(num_feature * 4) for _ in range(num_residual)]
        )

        self.up_blocks = nn.ModuleList([
            # 当stride > 1时，Conv2d将多个输入形状映射到相同的输出形状。output_padding通过在图像的右侧和底部分别增加计算出的输出形状来解决这种模糊性。
            # output_padding应该取值为stride - 1，这样就能够满足输入特征图大小/输出特征图大小 = stride，需要注意的这是当kernel_size是奇数时。
            ConvBlock(num_feature * 4, num_feature * 2, down=False, kernel_size=3, stride=2, padding=1,
                      output_padding=1),
            ConvBlock(num_feature * 2, num_feature * 1, down=False, kernel_size=3, stride=2, padding=1,
                      output_padding=1)
        ])

        self.out = nn.Sequential(
            nn.Conv2d(num_feature, img_channels, 7, 1, 3, padding_mode="reflect"),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.initial(x)
        for layer in self.down_blocks:
            x = layer(x)
        x = self.residual_block(x)
        for layer in self.up_blocks:
            x = layer(x)
        return self.out(x)


def test():
    img_channel = 3
    img_size = 256
    x = torch.randn((5, img_channel, img_size, img_size))
    gen = Generator(img_channel)
    pred = gen(x)
    print(pred.shape)


if __name__ == '__main__':
    test()
