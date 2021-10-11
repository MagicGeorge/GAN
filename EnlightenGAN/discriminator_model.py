import torch
import numpy as np
from torch import nn


def define_D(input_nc, ndf, n_layer_D, use_sigmoid):
    net_D = NoNormDiscriminator(input_nc, ndf, n_layer_D, use_sigmoid)
    return net_D


class NoNormDiscriminator(nn.Module):
    def __init__(self, input_nc=3, ndf=64, n_layers=4, use_sigmoid=True):  # ndf: number of filters in the discriminator
        super(NoNormDiscriminator, self).__init__()

        kw = 4
        padw = int(np.ceil((kw - 1) / 2))
        self.initial = nn.Sequential(
            nn.Conv2d(input_nc, ndf, kernel_size=(kw, kw), stride=(2, 2), padding=(padw, padw)),
            nn.LeakyReLU(0.2, inplace=True)
        )

        layers = []
        nf_mult = 1  # out_channel参数所需乘以的系数
        nf_mult_prev = 1  # in_channel参数所需乘以的系数
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            layers += [
                nn.Conv2d(nf_mult_prev * ndf, nf_mult * ndf, kernel_size=(kw, kw), stride=(2, 2), padding=(padw, padw)),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        layers += [
            nn.Conv2d(nf_mult_prev * ndf, nf_mult * ndf, kernel_size=(kw, kw), stride=(1, 1), padding=(padw, padw)),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf_mult * ndf, 1, kernel_size=(kw, kw), stride=(1, 1), padding=(padw, padw))
        ]

        if use_sigmoid:
            layers += [nn.Sigmoid()]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.initial(x)
        return self.model(x)


if __name__ == '__main__':
    net_D = define_D(3, 64, 5, True)
    input_x = torch.randn(1, 3, 256, 256)
    out = net_D(input_x)
    print(net_D)
    print(out.shape)
