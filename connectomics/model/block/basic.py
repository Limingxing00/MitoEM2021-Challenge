import os, sys
import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from ..norm import *


# common layers
def get_functional_act(mode='relu'):
    activation_dict = {
        'relu': F.relu_,
        'tanh': torch.tanh,
        'elu': F.elu_,
        'sigmoid': torch.sigmoid,
        'softmax': lambda x: F.softmax(x, dim=1),
        'none': lambda x: x,
    }
    return activation_dict[mode]


def get_layer_act(mode=''):
    if mode == '':
        return []
    elif mode == 'relu':
        return [nn.ReLU(inplace=True)]
    elif mode == 'elu':
        return [nn.ELU(inplace=True)]
    elif mode[:5] == 'leaky':
        # 'leaky0.2' 
        return [nn.LeakyReLU(inplace=True, negative_slope=float(mode[5:]))]
    raise ValueError('Unknown activation layer option {}'.format(mode))


def get_layer_norm(out_planes, norm_mode='', dim=2):
    if norm_mode == '':
        return []
    elif norm_mode == 'bn':
        if dim == 1:
            return [SynchronizedBatchNorm1d(out_planes)]
        elif dim == 2:
            return [SynchronizedBatchNorm2d(out_planes)]
        elif dim == 3:
            return [SynchronizedBatchNorm3d(out_planes)]
    elif norm_mode == 'abn':
        if dim == 1:
            return [nn.BatchNorm1d(out_planes)]
        elif dim == 2:
            return [nn.BatchNorm2d(out_planes)]
        elif dim == 3:
            return [nn.BatchNorm3d(out_planes)]
    elif norm_mode == 'in':
        if dim == 1:
            return [nn.InstanceNorm1d(out_planes)]
        elif dim == 2:
            return [nn.InstanceNorm2d(out_planes)]
        elif dim == 3:
            return [nn.InstanceNorm3d(out_planes)]
    elif norm_mode == 'bin':
        if dim == 1:
            return [BatchInstanceNorm1d(out_planes)]
        elif dim == 2:
            return [BatchInstanceNorm2d(out_planes)]
        elif dim == 3:
            return [BatchInstanceNorm3d(out_planes)]
    raise ValueError('Unknown normalization norm option {}'.format(mode))


# conv basic blocks
def conv2d_norm_act(in_planes, out_planes, kernel_size=(3, 3), stride=1,
                    dilation=(1, 1), padding=(1, 1), bias=True, pad_mode='rep', norm_mode='', act_mode='',
                    return_list=False):
    if isinstance(padding, int):
        pad_mode = pad_mode if padding != 0 else 'zeros'
    else:
        pad_mode = pad_mode if max(padding) != 0 else 'zeros'

    if pad_mode in ['zeros', 'circular']:
        layers = [nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size,
                            stride=stride, padding=padding, padding_mode=pad_mode, dilation=dilation, bias=bias)]
    elif pad_mode == 'rep':
        # the size of the padding should be a 6-tuple        
        padding = tuple([x for x in padding for _ in range(2)][::-1])
        layers = [nn.ReplicationPad2d(padding),
                  nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size,
                            stride=stride, padding=0, dilation=dilation, bias=bias)]
    else:
        raise ValueError('Unknown padding option {}'.format(mode))
    # layers += get_layer_norm(out_planes, norm_mode)
    layers += get_layer_act(act_mode)
    if return_list:
        return layers
    else:
        return nn.Sequential(*layers)


def conv3d_norm_act(in_planes, out_planes, kernel_size=(3, 3, 3), stride=1,
                    dilation=(1, 1, 1), padding=(1, 1, 1), bias=True, pad_mode='rep', norm_mode='', act_mode='',
                    return_list=False):
    if isinstance(padding, int):
        pad_mode = pad_mode if padding != 0 else 'zeros'
    else:
        pad_mode = pad_mode if max(padding) != 0 else 'zeros'

    if pad_mode in ['zeros', 'circular']:
        layers = [nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size,
                            stride=stride, padding=padding, padding_mode=pad_mode, dilation=dilation, bias=bias)]
    elif pad_mode == 'rep':
        # the size of the padding should be a 6-tuple        
        padding = tuple([x for x in padding for _ in range(2)][::-1])
        layers = [nn.ReplicationPad3d(padding),
                  nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size,
                            stride=stride, padding=0, dilation=dilation, bias=bias)]
    else:
        raise ValueError('Unknown padding option {}'.format(mode))

    # layers += get_layer_norm(out_planes, norm_mode, 3)
    layers += get_layer_act(act_mode)
    if return_list:
        return layers
    else:
        return nn.Sequential(*layers)


# -----------------------------------------------------------------------------------------
# Non-local module
# https://github.com/AlexHex7/Non-local_pytorch/blob/master/lib/non_local_concatenation.py
# error: GPU memory
# -----------------------------------------------------------------------------------------

class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = max_pool_layer

    def forward(self, x, return_nl_map=False):
        """
        :param x: (b, c, t, h, w)
        :param return_nl_map: if True return z, nl_map, else only return z.
        :return:
        """

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)

        g_x = g_x.permute(0, 2, 1)

        theta_x = x.view(batch_size, self.in_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)

        if self.sub_sample:
            phi_x = self.phi(x).view(batch_size, self.in_channels, -1)
        else:
            phi_x = x.view(batch_size, self.in_channels, -1)

        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        # if self.store_last_batch_nl_map:
        #     self.nl_map = f_div_C

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        if return_nl_map:
            return z, f_div_C
        return z


class NONLocalBlock1D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock1D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=1, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class NONLocalBlock2D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock2D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=2, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class NONLocalBlock3D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock3D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=3, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


# ----------------------------------------------------------------------
# Non-local CC

# import torch.autograd as autograd
# import torch.cuda.comm as comm
# import torch.nn.functional as F
# from torch.autograd.function import once_differentiable
# from torch.utils.cpp_extension import load
# import os, time
# import functools
#
# curr_dir = os.path.dirname(os.path.abspath(__file__))
# _src_path = os.path.join(curr_dir, "src")
# _build_path = os.path.join(curr_dir, "build")
# os.makedirs(_build_path, exist_ok=True)
# rcca = load(name="rcca",
#             extra_cflags=["-O3"],
#             build_directory=_build_path,
#             verbose=True,
#             sources=[os.path.join(_src_path, f) for f in [
#                 "lib_cffi.cpp", "ca.cu"
#             ]],
#             extra_cuda_cflags=["--expt-extended-lambda"])
#
#
# def _check_contiguous(*args):
#     if not all([mod is None or mod.is_contiguous() for mod in args]):
#         raise ValueError("Non-contiguous input")
#
#
# class CA_Weight(autograd.Function):
#     @staticmethod
#     def forward(ctx, t, f):
#         # Save context
#         n, c, d, h, w = t.size()
#         size = (n, h + w + d - 1, d, h, w)
#         weight = torch.zeros(size, dtype=t.dtype, layout=t.layout, device=t.device)
#
#         rcca.ca_forward_cuda(t, f, weight)
#
#         # Output
#         ctx.save_for_backward(t, f)
#
#         return weight
#
#     @staticmethod
#     @once_differentiable
#     def backward(ctx, dw):
#         t, f = ctx.saved_tensors
#
#         dt = torch.zeros_like(t)
#         df = torch.zeros_like(f)
#
#         rcca.ca_backward_cuda(dw.contiguous(), t, f, dt, df)
#
#         _check_contiguous(dt, df)
#
#         return dt, df
#
#
# class CA_Map(autograd.Function):
#     @staticmethod
#     def forward(ctx, weight, g):
#         # Save context
#         out = torch.zeros_like(g)
#         rcca.ca_map_forward_cuda(weight, g, out)
#
#         # Output
#         ctx.save_for_backward(weight, g)
#
#         return out
#
#     @staticmethod
#     @once_differentiable
#     def backward(ctx, dout):
#         weight, g = ctx.saved_tensors
#
#         dw = torch.zeros_like(weight)
#         dg = torch.zeros_like(g)
#
#         rcca.ca_map_backward_cuda(dout.contiguous(), weight, g, dw, dg)
#
#         _check_contiguous(dw, dg)
#
#         return dw, dg
#
#
# ca_weight = CA_Weight.apply
# ca_map = CA_Map.apply
#
#
# class CrissCrossAttention(nn.Module):
#     """ Criss-Cross Attention Module
#     ca = CrissCrossAttention(256).cuda()
#     x = torch.zeros(1, 8, 10, 10).cuda() + 1
#     y = torch.zeros(1, 8, 10, 10).cuda() + 2
#     z = torch.zeros(1, 64, 10, 10).cuda() + 3
#     out = ca(x, y, z)
#     """
#
#     def __init__(self, in_dim):
#         super(CrissCrossAttention, self).__init__()
#         self.chanel_in = in_dim
#
#         self.query_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim // 2, kernel_size=1)
#         self.key_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim // 2, kernel_size=1)
#         self.value_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
#         self.gamma = nn.Parameter(torch.zeros(1))
#
#     def forward(self, x):
#         proj_query = self.query_conv(x)
#         proj_key = self.key_conv(x)
#         proj_value = self.value_conv(x)
#
#         energy = ca_weight(proj_query, proj_key)
#         attention = F.softmax(energy, 1)
#         out = ca_map(attention, proj_value)
#         out = self.gamma * out + x
#         return out
