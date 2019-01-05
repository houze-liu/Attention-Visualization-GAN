import torch.nn as nn
import numpy as np
from spectral import  SpectralNorm
from self_attention import SelfAttention


class ResidualBlock(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            SpectralNorm(nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False)),
            nn.ReLU(inplace=True),
            SpectralNorm(nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False)))

    def forward(self, x):
        return x + self.main(x)


class Generator(nn.Module):
    '''Generator network that grows progressively?'''
    def __init__(self, image_size=64, z_dim=100, conv_dim=64):
        # no use cause network needs to be build first and can not change duing forward
        super(Generator, self).__init__()
        self.imsize = image_size
        self.watch_list1 = [0] # a list used to store attention map
        self.watch_list2 = [0]

        layers = []

        repeat_num = int(np.log2(self.imsize)) - 3  # 3
        mult = 2 ** repeat_num  # 8; multiplier to conv_dim
        curr_dim = z_dim # initial dim equals z_dim
        tar_dim = conv_dim * mult # initial tar_dim

        layers.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, conv_dim * mult, 4)))
        layers.append(nn.BatchNorm2d(conv_dim * mult)) # batch norm before none-linearity
        layers.append(nn.ReLU())
        curr_dim = tar_dim
        tar_dim = int(tar_dim / 2)

        for i in range(repeat_num):
            layers.append(
                SpectralNorm(nn.ConvTranspose2d(curr_dim, tar_dim, 4,2,1))) # transpose
            layers.append(nn.BatchNorm2d(tar_dim))
            layers.append(nn.ReLU())
            curr_dim = tar_dim
            tar_dim = int(tar_dim / 2)

            if curr_dim == 64:
                self.attn1 = SelfAttention(64, self.watch_list1)
                layers.append(self.attn1)
            if curr_dim == 128:
                self.attn2 = SelfAttention(128, self.watch_list2)
                layers.append(self.attn2)

        layers.append(nn.ConvTranspose2d(curr_dim, 3, kernel_size=4, stride=2, padding=1))
        layers.append(nn.Tanh())

        self.main = nn.Sequential(*layers)

    def forward(self, z):
        z = z.view(z.size(0), z.size(1), 1, 1)
        out = self.main(z)

        return out


class Discriminator(nn.Module):
    '''Discriminator structure'''
    def __init__(self, conv_dim=64, watch_on=False):
        super(Discriminator, self).__init__()
        self.watch_list1 = [0]
        self.watch_list2 = [0]
        layers = []

        curr_dim = 3  # initial dim equals z_dim
        tar_dim = conv_dim  # initial tar_dim

        for i in range(4):
            layers.append(
                SpectralNorm(nn.Conv2d(curr_dim, tar_dim, 4, 2, 1)))
            layers.append(nn.BatchNorm2d(tar_dim))
            layers.append(nn.LeakyReLU(0.1))
            curr_dim = tar_dim
            tar_dim = curr_dim * 2

            if curr_dim == 256:
                layers.append(SelfAttention(256, self.watch_list1))
            if curr_dim == 512:
                layers.append(SelfAttention(512, self.watch_list2))
        layers.append(nn.Conv2d(curr_dim, 1, 4))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        out = self.main(x)

        return out.squeeze()
