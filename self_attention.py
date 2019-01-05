import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F


def init_conv(conv):
    init.xavier_uniform_(conv.weight) # xavier for weights and 0 for biases
    if conv.bias is not None:
        conv.bias.data.zero_()


class SelfAttention(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, watch_list):
        super(SelfAttention, self).__init__()
        self.chanel_in = in_dim
        self.watch_list = watch_list

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

        init_conv(self.query_conv)
        init_conv(self.key_conv)
        init_conv(self.value_conv)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height) compare pixel by pixel
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X N X C
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x N
        energy = torch.bmm(proj_query, proj_key)  # B X (N) X (N);
        attention = self.softmax(energy)
        # channel can be seen as correlation to postions and reduced
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1)) # minibatch matrix product
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x

        self.watch_list.pop() # pop out last attn so that memory can be saved
        self.watch_list.append(attention)
        return out