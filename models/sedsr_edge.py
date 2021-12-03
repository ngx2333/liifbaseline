# modified from: https://github.com/thstkdgus35/EDSR-PyTorch

import math
from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import register
from .pac import PacConv2d


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)


def pac_conv(in_channels, out_channels, kernel_size, bias=True):
    return PacConv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)


class MeanShift(nn.Conv2d):
    def __init__(
            self, rgb_range,
            rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class EdgeResBlock(nn.Module):
    def __init__(
            self, conv, n_feats, kernel_size,
            bias=True, bn=False, act=nn.ReLU(False), res_scale=1):

        super(EdgeResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(pac_conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        m_edge = []
        for i in range(2):
            m_edge.append(default_conv(
                n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m_edge.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m_edge.append(act)

        self.sr_body = nn.Sequential(*m)
        self.edge_body = nn.Sequential(*m_edge)
        self.res_scale = res_scale

    def forward(self, x, edge):

        res_sr, res_edge = x, edge
        for i, (sr_lay, edge_lay) in enumerate(zip(self.sr_body, self.edge_body)):
            if i == 1:
                res_sr = sr_lay(res_sr)
                res_edge = edge_lay(res_edge)
            else:
                res_sr = sr_lay(res_sr, res_edge)
                res_edge = edge_lay(res_edge)

        res_sr = res_sr.mul(self.res_scale)
        res_edge = res_edge.mul(self.res_scale)
        res_sr = x+res_sr
        res_edge = edge+res_edge
        return res_sr, res_edge


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(False))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(False))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


url = {
    'r16f64x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x2-1bc95232.pt',
    'r16f64x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x3-abf2a44e.pt',
    'r16f64x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x4-6b446fab.pt',
    'r32f256x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x2-0edfb8a3.pt',
    'r32f256x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x3-ea3ef2c6.pt',
    'r32f256x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x4-4f62e9ef.pt'
}


class EDSR(nn.Module):
    def __init__(self, args, conv=default_conv):
        super(EDSR, self).__init__()
        self.args = args
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        scale = args.scale[0]
        act = nn.ReLU(False)
        url_name = 'r{}f{}x{}'.format(n_resblocks, n_feats, scale)
        if url_name in url:
            self.url = url[url_name]
        else:
            self.url = None
        self.sub_mean = MeanShift(args.rgb_range)
        self.add_mean = MeanShift(args.rgb_range, sign=1)

        # define head module
        sr_m_head = [conv(args.n_colors, n_feats, kernel_size)]
        edge_m_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        m_body = [
            EdgeResBlock(
                pac_conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblocks)
        ]

        self.sr_body_conv = conv(n_feats, n_feats, kernel_size)
        self.edge_body_conv = conv(n_feats, n_feats, kernel_size)
        # m_body.append(conv(n_feats, n_feats, kernel_size))

        self.sr_head = nn.Sequential(*sr_m_head)
        self.edge_head = nn.Sequential(*edge_m_head)

        self.body = nn.Sequential(*m_body)

        if args.no_upsampling:
            self.out_dim = n_feats
        else:
            self.out_dim = args.n_colors
            # define tail module
            edge_m_tail = [
                Upsampler(default_conv, scale, n_feats, act=False),
                default_conv(n_feats, args.out_colors, kernel_size)
            ]
            self.edge_tail = nn.Sequential(*edge_m_tail)

            sr_m_tail = [
                Upsampler(default_conv, scale, n_feats, act=False),
                pac_conv(n_feats, args.n_colors, kernel_size)
            ]
            self.sr_tail = nn.Sequential(*sr_m_tail)

    def forward(self, x):
        #x = self.sub_mean(x)
        sr_x = self.sr_head(x)
        edge_x = self.edge_head(x)
        res_sr, res_edge = sr_x, edge_x
        for lay in self.body:
            res_sr, res_edge = lay(sr_x, edge_x)
        res_sr = self.sr_body_conv(res_sr)
        res_edge = self.edge_body_conv(res_edge)
        # res_sr = sr_x+res_sr
        # res_edge = edge_x+res_edge

        sr_x = res_sr
        edge_x = res_edge

        if not self.args.no_upsampling:
            for i, (sr_lay, edge_lay) in enumerate(zip(self.sr_tail, self.edge_tail)):
                if i == 0:
                    sr_x = sr_lay[0](sr_x)
                    edge_x = edge_lay[0](edge_x)
                    sr_x = sr_lay[1](sr_x)
                    edge_x = edge_lay[1](edge_x)
                else:
                    sr_x = sr_lay(sr_x, edge_x)
                    edge_x = edge_lay(edge_x)
        #x = self.add_mean(x)
        edge_x = torch.sigmoid(edge_x)
        return sr_x, edge_x

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))


@register('sedsr-edge')
def make_edsr_baseline(n_resblocks=16, n_feats=64, res_scale=1,
                       scale=2, no_upsampling=False, rgb_range=1):
    args = Namespace()
    args.n_resblocks = n_resblocks
    args.n_feats = n_feats
    args.res_scale = res_scale

    args.scale = [scale]
    args.no_upsampling = no_upsampling

    args.rgb_range = rgb_range
    args.n_colors = 3
    args.out_colors = 1

    return EDSR(args)


if __name__ == '__main__':
    model = make_edsr_baseline()
    # print(model)
    x = torch.rand([1, 3, 256, 256])
    sr_y, edge_y = model(x)
    print(sr_y.shape)
    print(edge_y.shape)

    # block=EdgeResBlock(conv=default_conv,n_feats=3,kernel_size=3)
    # y=block(x,x)
    # print(block)
    # 要卷来实验室卷
