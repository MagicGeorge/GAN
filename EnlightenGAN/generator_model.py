import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from options import TrainOptions


def define_G(opt=None):
    net_G = SelfAttentionUnet(opt)
    return net_G


def pad_tensor(x):
    """
    如果图像大小不能进行四次下采样，则需要事先padding，这里采用Reflection方式进行填充
    :param x:
    """
    height_org = x.shape[2]
    width_org = x.shape[3]
    divide = 16  # 因为Generator的encoder结构需要降采样4次，2^4=16。即图像大小至少是16的倍数

    height_res = height_org % divide
    width_res = width_org % divide
    if height_res != 0 or width_res != 0:
        if height_res != 0:
            height_div = divide - height_res
            pad_top = int(height_div / 2)  # int()的作用是取整，无论正数还是负数，直接去掉小数点之后的部分
            pad_bottom = int(height_div - pad_top)
        else:
            pad_top = 0
            pad_bottom = 0

        if width_res != 0:
            width_div = divide - width_res
            pad_left = int(width_div / 2)
            pad_right = int(width_div - pad_left)
        else:
            pad_left = 0
            pad_right = 0
        padding = nn.ReflectionPad2d((pad_left, pad_right, pad_top, pad_bottom))
        x = padding(x)
    else:
        pad_left = 0
        pad_right = 0
        pad_top = 0
        pad_bottom = 0

    height = x.shape[2]
    width = x.shape[3]
    assert width % divide == 0, 'width cant divided by stride'
    assert height % divide == 0, 'height cant divided by stride'

    return x, pad_left, pad_right, pad_top, pad_bottom


def pad_tensor_back(x, pad_left, pad_right, pad_top, pad_bottom):
    """
    还原回原始图像大小
    :param x:
    :param pad_left:
    :param pad_right:
    :param pad_top:
    :param pad_bottom:
    """
    height, width = x.shape[2], x.shape[3]
    return x[:, :, pad_top: height - pad_bottom, pad_left: width - pad_right]


class ConvBlock(nn.Module):
    def __init__(self, input_nc, output_nc, use_norm):
        super(ConvBlock, self).__init__()

        layers = [
            nn.Conv2d(input_nc, output_nc, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, True)
        ]
        if use_norm:
            layers += [nn.BatchNorm2d(output_nc)]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UpsampleBlock(nn.Module):
    def __init__(self, input_nc, output_nc, use_transpose_conv):
        super(UpsampleBlock, self).__init__()

        # 上采样方式可以选择Conv+upsample或者ConvTranspose
        if use_transpose_conv:
            self.model = nn.ConvTranspose2d(input_nc, output_nc, kernel_size=(2, 2), stride=(2, 2))
        else:
            self.model = nn.Sequential(
                nn.Conv2d(input_nc, output_nc, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            )

    def forward(self, x):
        return self.model(x)


class SelfAttentionUnet(nn.Module):
    def __init__(self, opt):
        super(SelfAttentionUnet, self).__init__()

        self.opt = opt

        if self.opt.self_attention:
            self.conv1_1 = ConvBlock(4, 32, self.opt.use_norm)
            self.downsample_1 = nn.AvgPool2d(2)  # 调整注意力图的大小，以适应每个skip connection的特征图大小
            self.downsample_2 = nn.AvgPool2d(2)  # 因为有5次skip，需要做4次调整
            self.downsample_3 = nn.AvgPool2d(2)
            self.downsample_4 = nn.AvgPool2d(2)
        else:
            self.conv1_1 = ConvBlock(3, 32, self.opt.use_norm)

        #  Encoder
        self.conv1_2 = ConvBlock(32, 32, self.opt.use_norm)
        self.max_pool1 = nn.MaxPool2d(2)

        self.conv2_1 = ConvBlock(32, 64, self.opt.use_norm)
        self.conv2_2 = ConvBlock(64, 64, self.opt.use_norm)
        self.max_pool2 = nn.MaxPool2d(2)

        self.conv3_1 = ConvBlock(64, 128, self.opt.use_norm)
        self.conv3_2 = ConvBlock(128, 128, self.opt.use_norm)
        self.max_pool3 = nn.MaxPool2d(2)

        self.conv4_1 = ConvBlock(128, 256, self.opt.use_norm)
        self.conv4_2 = ConvBlock(256, 256, self.opt.use_norm)
        self.max_pool4 = nn.MaxPool2d(2)

        self.conv5_1 = ConvBlock(256, 512, self.opt.use_norm)
        self.conv5_2 = ConvBlock(512, 512, self.opt.use_norm)

        #  Decoder
        self.upsample1 = UpsampleBlock(512, 256, self.opt.use_transpose_conv)

        self.conv6_1 = ConvBlock(512, 256, self.opt.use_norm)
        self.conv6_2 = ConvBlock(256, 256, self.opt.use_norm)
        self.upsample2 = UpsampleBlock(256, 128, self.opt.use_transpose_conv)

        self.conv7_1 = ConvBlock(256, 128, self.opt.use_norm)
        self.conv7_2 = ConvBlock(128, 128, self.opt.use_norm)
        self.upsample3 = UpsampleBlock(128, 64, self.opt.use_transpose_conv)

        self.conv8_1 = ConvBlock(128, 64, self.opt.use_norm)
        self.conv8_2 = ConvBlock(64, 64, self.opt.use_norm)
        self.upsample4 = UpsampleBlock(64, 32, self.opt.use_transpose_conv)

        self.conv9_1 = ConvBlock(64, 32, self.opt.use_norm)
        self.conv9_2 = ConvBlock(32, 32, self.opt.use_norm)
        self.conv_last = nn.Conv2d(32, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        if self.opt.tanh:
            self.tanh = nn.Tanh()

    def forward(self, input, gray):
        flag = 0
        if input.shape[3] > 2200:  # 如果图像大小超过2200, 则缩小图像至1/4大小, 以加快训练
            avg = nn.AvgPool2d(2)
            input = avg(input)
            gray = avg(gray)
            flag = 1  # 用于标记，以便在return之前恢复图像大小
        input, pad_left, pad_right, pad_top, pad_bottom = pad_tensor(input)
        gray, pad_left, pad_right, pad_top, pad_bottom = pad_tensor(gray)

        if self.opt.self_attention:
            gray_2 = self.downsample_1(gray)
            gray_3 = self.downsample_2(gray_2)
            gray_4 = self.downsample_3(gray_3)
            gray_5 = self.downsample_4(gray_4)
            x = self.conv1_1(torch.cat((input, gray), dim=1))
        else:
            x = self.conv1_1(input)

        conv1 = self.conv1_2(x)
        x = self.max_pool1(conv1)

        conv2 = self.conv2_2(self.conv2_1(x))
        x = self.max_pool2(conv2)

        conv3 = self.conv3_2(self.conv3_1(x))
        x = self.max_pool3(conv3)

        conv4 = self.conv4_2(self.conv4_1(x))
        x = self.max_pool4(conv4)

        x = self.conv5_1(x)
        # a*b相当于torch.mul(对位相乘); a@b相当于torch.matmul
        x = x * gray_5 if self.opt.self_attention else x  # bottleneck层没有skip connection, 仍然施加了注意力图的自我正则
        x = self.upsample1(self.conv5_2(x))

        conv4 = conv4 * gray_4 if self.opt.self_attention else conv4
        x = torch.cat((x, conv4), dim=1)
        x = self.upsample2(self.conv6_2(self.conv6_1(x)))

        conv3 = conv3 * gray_3 if self.opt.self_attention else conv3
        x = torch.cat((x, conv3), dim=1)
        x = self.upsample3(self.conv7_2(self.conv7_1(x)))

        conv2 = conv2 * gray_2 if self.opt.self_attention else conv2
        x = torch.cat((x, conv2), dim=1)
        x = self.upsample4(self.conv8_2(self.conv8_1(x)))

        conv1 = conv1 * gray if self.opt.self_attention else conv1
        x = torch.cat((x, conv1), dim=1)
        x = self.conv_last(self.conv9_2(self.conv9_1(x)))

        if self.opt.mult_attention:
            x = x * gray

        if self.opt.tanh:
            x = self.tanh(x)

        skip = True if self.opt.residual_skip > 0 else False  # 指示模型是否做最外层的残差学习
        if skip:
            if self.opt.linear_stretch:
                input = (input - torch.min(input)) / (torch.max(input) - torch.min(input))
                output = x + input * self.opt.residual_skip
                output = output * 2 - 1  # (x - 0.5) / 0.5
            else:
                output = x + input * self.opt.residual_skip
        else:
            output = x

        output = pad_tensor_back(output, pad_left, pad_right, pad_top, pad_bottom)
        # x = pad_tensor_back(x, pad_left, pad_right, pad_top, pad_bottom)

        if flag == 1:
            output = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=True)  # align_corners参数
            # gray = F.interpolate(gray, scale_factor=2, mode='bilinear', align_corners=True)

        return output


if __name__ == '__main__':
    x = torch.randn(32, 3, 256, 256)
    gray = torch.randn(32, 1, 256, 256)
    sig = nn.Sigmoid()
    gray = sig(gray)
    opt = TrainOptions().parse()
    net_G = define_G(opt)
    output = net_G(x, gray)

    print(output.shape)
