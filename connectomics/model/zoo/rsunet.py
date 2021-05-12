# -*- coding: utf-8 -*-
# @Time    : 2021/1/16 9:17
# @Author  : Mingxing Li
# @FileName: rsunet.py
# @Software: PyCharm
"""
Symmetric (Partial) 3D U-Net.
Residual skip connections. (Optional)

Kisuk Lee <kisuklee@mit.edu>, 2017-2018
Nicholas Turner <nturner@cs.princeton.edu>, 2017
Chang Chen <changc@mail.ustc.edu.cn>, 2020
"""

import math
import collections
from itertools import repeat
from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F
from .p3d import PartialConv3d


def _ntuple(n):
    """ Copied from PyTorch source code (https://github.com/pytorch). """

    def parse(x):
        if isinstance(x, collections.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


_triple = _ntuple(3)


def pad_size(kernel_size, mode):
    assert mode in ['valid', 'same', 'full']
    ks = _triple(kernel_size)
    if mode == 'valid':
        pad = (0, 0, 0)
    elif mode == 'same':
        assert all([x % 2 for x in ks])
        pad = tuple(x // 2 for x in ks)
    elif mode == 'full':
        pad = tuple(x - 1 for x in ks)
    return pad


def batchnorm(out_channels, use_bn, momentum=0.001):
    if use_bn:
        layer = nn.BatchNorm3d(out_channels, eps=1e-05, momentum=momentum)
    else:
        layer = lambda x: x
    return layer


def residual_sum(x, skip, residual):
    return x + skip if residual else x


class Conv(nn.Module):
    """ 3D convolution w/ MSRA init. """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        nn.init.kaiming_normal_(self.conv.weight)
        if bias:
            nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        return self.conv(x)


class PConv(nn.Module):
    """ 3D partial convolution w/ MSRA init. """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(PConv, self).__init__()
        self.conv = PartialConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias,
                                  return_mask=True, multi_channel=True)
        nn.init.kaiming_normal_(self.conv.weight)
        if bias:
            nn.init.constant_(self.conv.bias, 0)

    def forward(self, x, mask_in):
        return self.conv(x, mask_in)


class ConvT(nn.Module):
    """ 3D convolution transpose w/ MSRA init. """

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True):
        super(ConvT, self).__init__()
        self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                       bias=bias)
        nn.init.kaiming_normal_(self.conv.weight)
        if bias:
            nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        return self.conv(x)


class ConvMod(nn.Module):
    """ Convolution module. """

    def __init__(self, in_channels, out_channels, kernel_size, activation=F.elu, residual=True):
        super(ConvMod, self).__init__()
        # Convolution params.
        ks = _triple(kernel_size)
        st = (1, 1, 1)
        pad = pad_size(ks, 'same')
        bias = True

        # Convolutions.
        # self.conv1 = Conv(in_channels,  out_channels, ks, st, pad, bias)
        self.conv1 = Conv(in_channels, out_channels, (1, 3, 3), st, (0, 1, 1), bias)
        self.conv2 = Conv(out_channels, out_channels, ks, st, pad, bias)
        self.conv3 = Conv(out_channels, out_channels, ks, st, pad, bias)

        # Activation function.
        self.activation = activation
        # Residual skip connection.
        self.residual = residual

    def forward(self, x):
        # Conv 1.
        x = self.conv1(x)
        x = self.activation(x)
        skip = x
        # Conv 2.
        x = self.conv2(x)
        x = self.activation(x)
        # Conv 3.
        x = self.conv3(x)
        x = residual_sum(x, skip, self.residual)
        return self.activation(x)


class PConvMod(nn.Module):
    """ Partial convolution module. """

    def __init__(self, in_channels, out_channels, kernel_size, activation=F.elu):
        super(PConvMod, self).__init__()
        # Convolution params.
        ks = _triple(kernel_size)
        st = (1, 1, 1)
        pad = pad_size(ks, 'same')
        bias = True

        # Partial convolutions (return_mask, multi_channel).
        self.conv1 = PConv(in_channels, out_channels, ks, st, pad, bias)
        self.conv2 = PConv(out_channels, out_channels, ks, st, pad, bias)
        self.activation = activation

    def forward(self, x, mask_in):
        # Conv 1.
        x, mask = self.conv1(x, mask_in)
        x = self.activation(x)
        # Conv 2.
        x, mask = self.conv2(x, mask)
        x = self.activation(x)
        # assert mask
        return x


class BilinearUp(nn.Module):
    """ Caffe style bilinear upsampling.
        Currently everything's hardcoded and only supports upsampling factor of 2. """

    def __init__(self, in_channels, out_channels, factor=(1, 2, 2)):
        super(BilinearUp, self).__init__()
        assert in_channels == out_channels
        self.groups = in_channels
        self.factor = factor
        self.kernel_size = [(2 * f) - (f % 2) for f in self.factor]
        self.padding = [int(math.ceil((f - 1) / 2.0)) for f in factor]
        self.init_weights()

    def init_weights(self):
        weight = torch.Tensor(self.groups, 1, *self.kernel_size)
        width = weight.size(-1)
        hight = weight.size(-2)
        assert width == hight
        f = float(math.ceil(width / 2.0))
        c = float(width - 1) / (2.0 * f)
        for w in range(width):
            for h in range(hight):
                weight[..., h, w] = (1 - abs(w / f - c)) * (1 - abs(h / f - c))
        self.register_buffer('weight', weight)  # fixed

    def forward(self, x):
        return F.conv_transpose3d(x, self.weight, stride=self.factor, padding=self.padding, groups=self.groups)


class UpsampleMod(nn.Module):
    """ Transposed Convolution module. """

    def __init__(self, in_channels, out_channels, up=(1, 2, 2), mode='bilinear', activation=F.elu):
        super(UpsampleMod, self).__init__()
        # Convolution params.
        ks = (1, 1, 1)
        st = (1, 1, 1)
        pad = (0, 0, 0)
        bias = True
        # Upsampling.
        if mode == 'bilinear':
            # self.up = nn.Upsample(scale_factor=up, mode='trilinear')
            self.up = BilinearUp(in_channels, in_channels, factor=up)
            self.conv = Conv(in_channels, out_channels, ks, st, pad, bias)
        elif mode == 'nearest':
            self.up = nn.Upsample(scale_factor=up, mode='nearest')
            self.conv = Conv(in_channels, out_channels, ks, st, pad, bias)
        elif mode == 'transpose':
            self.up = ConvT(in_channels, out_channels, kernel_size=up, stride=up, bias=bias)
            self.conv = lambda x: x
        else:
            assert False, "unknown upsampling mode {}".format(mode)
        self.activation = activation

    def forward(self, x, skip):
        x = self.up(x)
        x = self.conv(x)
        x = x + skip
        return self.activation(x)


class OutputMod(nn.Module):
    """ Output module. """

    def __init__(self, in_channels, out_channels, kernel_size, activation=torch.sigmoid):
        super(OutputMod, self).__init__()
        pad = pad_size(kernel_size, 'same')
        self.conv = Conv(in_channels, out_channels, kernel_size, stride=1, padding=pad, bias=True)
        self.activation = activation

    def forward(self, x):
        return self.activation(self.conv(x))


class RSUNet(nn.Module):
    """ Residual Symmetric U-Net (RSUNet).
    Args:
        in_channels (int): Number of input channels.
        depth (int): Depth/scale of U-Net.
        residual (bool, optional): Use residual skip connection?
        upsample (string, optional): Upsampling mode in ['bilinear', 'nearest', 'transpose']
        use_bn (bool, optional): Use batch normalization?
        momentum (float, optional): Momentum for batch normalization.
    """

    def __init__(self, nfeatures, in_channels=1, depth=None, residual=True, upsample='bilinear'):
        super(RSUNet, self).__init__()
        self.residual = residual
        self.upsample = upsample

        sizes = [(3, 3, 3)] * len(nfeatures)
        if depth == None:
            depth = len(nfeatures) - 1
        else:
            assert depth < len(nfeatures)
        self.depth = depth

        # Input feature embedding without batchnorm.
        embed_nin = nfeatures[0]
        self.embed_in = PConvMod(in_channels, embed_nin, 3)
        in_channels = embed_nin

        # Contracting/downsampling pathway.
        for d in range(depth):
            fs, ks = nfeatures[d], sizes[d]
            self.add_conv_mod(d, in_channels, fs, ks)
            self.add_max_pool(d + 1, fs)
            in_channels = fs

        # Bridge.
        fs, ks = nfeatures[depth], sizes[depth]
        self.add_conv_mod(depth, in_channels, fs, ks)
        in_channels = fs

        # Expanding/upsampling pathway.
        for d in reversed(range(depth)):
            fs, ks = nfeatures[d], sizes[d]
            self.add_upsample_mod('o', d, in_channels, fs)
            self.add_upsample_mod('c', d, in_channels, fs)
            in_channels = fs
            self.add_dconv_mod('o', d, in_channels, fs, ks)
            self.add_dconv_mod('c', d, in_channels, fs, ks)

        # Output feature embedding without batchnorm.
        self.embed_out1 = OutputMod(in_channels, 1, (1, 3, 3))
        self.embed_out2 = OutputMod(in_channels, 1, (1, 3, 3))

    def add_conv_mod(self, depth, in_channels, out_channels, kernel_size):
        name = 'convmod{}'.format(depth)
        module = ConvMod(in_channels, out_channels, kernel_size, residual=self.residual)
        self.add_module(name, module)

    def add_dconv_mod(self, mode, depth, in_channels, out_channels, kernel_size):
        name = 'dconvmod{}_{}'.format(depth, mode)
        module = ConvMod(in_channels, out_channels, kernel_size, residual=self.residual)
        self.add_module(name, module)

    def add_max_pool(self, depth, in_channels, down=(1, 2, 2)):
        name = 'maxpool{}'.format(depth)
        module = nn.MaxPool3d(down)
        self.add_module(name, module)

    def add_upsample_mod(self, mode, depth, in_channels, out_channels, up=(1, 2, 2)):
        name = 'upsample{}_{}'.format(depth, mode)
        module = UpsampleMod(in_channels, out_channels, up=up, mode=self.upsample)
        self.add_module(name, module)

    def forward(self, x, mask_in=None):
        # Input feature embedding without batchnorm.
        x = self.embed_in(x, mask_in)

        # Contracting/downsmapling pathway.
        skip = []
        for d in range(self.depth):
            convmod = getattr(self, 'convmod{}'.format(d))
            maxpool = getattr(self, 'maxpool{}'.format(d + 1))
            x = convmod(x)
            skip.append(x)
            x = maxpool(x)

        # Bridge.
        bridge = getattr(self, 'convmod{}'.format(self.depth))
        xx = bridge(x)

        # Expanding/upsampling pathway for object.
        x = xx
        for d in reversed(range(self.depth)):
            upsample = getattr(self, 'upsample{}_o'.format(d))
            dconvmod = getattr(self, 'dconvmod{}_o'.format(d))
            x = dconvmod(upsample(x, skip[d]))
        ox = x

        # Expanding/upsampling pathway for contour.
        x = xx
        for d in reversed(range(self.depth)):
            upsample = getattr(self, 'upsample{}_c'.format(d))
            dconvmod = getattr(self, 'dconvmod{}_c'.format(d))
            x = dconvmod(upsample(x, skip[d]))
        cx = x

        # Output without batchnorm.
        out_o = self.embed_out1(ox)
        out_c = self.embed_out2(cx)
        return torch.cat([out_o, out_c], dim=1)


if __name__ == '__main__':
    import numpy as np
    from torchsummary import summary
    # x = torch.Tensor(np.random.random((1, 1, 32, 256, 256)).astype(np.float32)).cuda()
    # mask = torch.Tensor(np.random.random((1, 1, 32, 256, 256)).astype(np.float32)).cuda()

    x = torch.Tensor(np.random.random((1, 1, 32, 256, 256)).astype(np.float32)).cuda()
    mask = torch.Tensor(np.random.random((1, 1, 32, 256, 256)).astype(np.float32)).cuda()

    model = RSUNet([16, 32, 48, 64, 80]).cuda()
    summary(model, input_size=(1, 32, 256, 256))
    with torch.no_grad():
        out_o, out_c = model(x, mask)
    print(out_o.shape, out_c.shape)
