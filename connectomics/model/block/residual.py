import os,sys
import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from .basic import *
from .ASPP import *
# 1. Residual blocks
# implemented with 2D conv
class residual_block_2d_c2(nn.Module):
    def __init__(self, in_planes, out_planes, projection=True, pad_mode='rep', norm_mode='bn', act_mode='elu'):
        super(residual_block_2d_c2, self).__init__()
        self.projection = projection
        self.conv = nn.Sequential(
            conv2d_norm_act( in_planes, out_planes, kernel_size=(3,3), padding=(1,1), pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode),
            conv2d_norm_act(out_planes, out_planes, kernel_size=(3,3), padding=(1,1), pad_mode=pad_mode, norm_mode=norm_mode)
        )
        self.projector = conv2d_norm_act(in_planes, out_planes, kernel_size=(1,1), padding=(0,0), norm_mode=norm_mode)
        self.act = get_layer_act(act_mode)[0]
        
    def forward(self, x):
        y = self.conv(x)
        if self.projection:
            y = y + self.projector(x)
        else:
            y = y + x
        y = self.act(y)
        return y  

# implemented with 3D conv
class residual_block_2d(nn.Module):
    """
    Residual Block 2D

    Args:
        in_planes (int): number of input channels.
        out_planes (int): number of output channels.
        projection (bool): projection of the input with a conv layer.
    """
    def __init__(self, in_planes, out_planes, projection=True, pad_mode='rep', norm_mode='bn', act_mode='elu'):
        super(residual_block_2d, self).__init__()
        self.projection = projection
        self.conv = nn.Sequential(
            conv3d_norm_act( in_planes, out_planes, kernel_size=(1,3,3), padding=(0,1,1), pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode),
            conv3d_norm_act(out_planes, out_planes, kernel_size=(1,3,3), padding=(0,1,1), pad_mode=pad_mode,norm_mode=norm_mode)
        )
        self.projector = conv3d_norm_act(in_planes, out_planes, kernel_size=(1,1,1), padding=(0,0,0), norm_mode=norm_mode)
        self.act = get_layer_act(act_mode)[0]
        
    def forward(self, x):
        y = self.conv(x)
        if self.projection:
            y = y + self.projector(x)
        else:
            y = y + x
        y = self.act(y)
        return y


class encoder_residual_block_3d_v1(nn.Module):
    """Residual Block 3D

    Args:
        in_planes (int): number of input channels.
        out_planes (int): number of output channels.
        projection (bool): projection of the input with a conv layer.
    """

    def __init__(self, in_planes, out_planes, projection=False, pad_mode='rep', norm_mode='bn', act_mode='elu'):
        super(encoder_residual_block_3d_v1, self).__init__()
        self.projection = projection
        self.conv1 = nn.Sequential(
            conv3d_norm_act(in_planes, out_planes, kernel_size=(3, 3, 3), padding=(1, 1, 1), pad_mode=pad_mode,
                            norm_mode=norm_mode, act_mode=act_mode),
            conv3d_norm_act(out_planes, out_planes, kernel_size=(3, 3, 3), padding=(1, 1, 1), pad_mode=pad_mode,
                            norm_mode=norm_mode)
        )
        self.conv2 = nn.Sequential(
            conv3d_norm_act(in_planes, out_planes, kernel_size=(1, 3, 3), padding=(0, 1, 1), pad_mode=pad_mode,
                            norm_mode=norm_mode, act_mode=act_mode),
            conv3d_norm_act(out_planes, out_planes, kernel_size=(1, 3, 3), padding=(0, 1, 1), pad_mode=pad_mode,
                            norm_mode=norm_mode)
        )

        self.fusion = conv3d_norm_act(out_planes, out_planes, kernel_size=(1, 1, 1), padding=(0, 0, 0), pad_mode=pad_mode,
                            norm_mode=norm_mode)

        self.projector = conv3d_norm_act(in_planes, out_planes, kernel_size=(1, 1, 1), padding=(0, 0, 0),
                                         norm_mode=norm_mode)
        self.act = get_layer_act(act_mode)[0]

    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.conv2(x)
        y = self.fusion(y1+y2)
        if self.projection:
            y = y + self.projector(x)
        else:
            y = y + x
        y = self.act(y)
        return y


class encoder_residual_block_3d_v2(nn.Module):
    """Residual Block 3D

    Args:
        in_planes (int): number of input channels.
        out_planes (int): number of output channels.
        projection (bool): projection of the input with a conv layer.
    """

    def __init__(self, in_planes, out_planes, projection=False, pad_mode='rep', norm_mode='bn', act_mode='elu'):
        super(encoder_residual_block_3d_v2, self).__init__()
        self.projection = projection
        self.conv1 = nn.Sequential(
            conv3d_norm_act(in_planes, out_planes, kernel_size=(3, 3, 3), padding=(1, 1, 1), pad_mode=pad_mode,
                            norm_mode=norm_mode, act_mode=act_mode),
            # conv3d_norm_act(out_planes, out_planes, kernel_size=(3, 3, 3), padding=(1, 1, 1), pad_mode=pad_mode,
            #                 norm_mode=norm_mode)
        )
        self.conv1_maspp = MASPP(inplanes=out_planes, reduction=3, ASPP_3D=True)

        self.conv2 = nn.Sequential(
            conv3d_norm_act(in_planes, out_planes, kernel_size=(1, 3, 3), padding=(0, 1, 1), pad_mode=pad_mode,
                            norm_mode=norm_mode, act_mode=act_mode),
            # conv3d_norm_act(out_planes, out_planes, kernel_size=(1, 3, 3), padding=(0, 1, 1), pad_mode=pad_mode,
            #                 norm_mode=norm_mode)
        )
        self.conv2_maspp = MASPP(inplanes=out_planes, reduction=3, ASPP_3D=False)

        self.fusion = conv3d_norm_act(out_planes, out_planes, kernel_size=(1, 1, 1), padding=(0, 0, 0), pad_mode=pad_mode,
                            norm_mode=norm_mode)

        self.projector = conv3d_norm_act(in_planes, out_planes, kernel_size=(1, 1, 1), padding=(0, 0, 0),
                                         norm_mode=norm_mode)
        self.act = get_layer_act(act_mode)[0]

    def forward(self, x):
        y1 = self.conv1(x)
        y1 = self.conv1_maspp(y1)
        y2 = self.conv2(x)
        y2 = self.conv2_maspp(y2)
        y = self.fusion(y1+y2)
        if self.projection:
            y = y + self.projector(x)
        else:
            y = y + x
        y = self.act(y)
        return y

class encoder_residual_block_3d_v3(nn.Module):
    """Residual Block 3D

    only save 3d and add (1, 6, 6) dilated conv.
    """

    def __init__(self, in_planes, out_planes, projection=False, pad_mode='rep', norm_mode='bn', act_mode='elu'):
        super(encoder_residual_block_3d_v3, self).__init__()
        self.projection = projection
        self.conv1 = nn.Sequential(
            conv3d_norm_act(in_planes, out_planes, kernel_size=(3, 3, 3), padding=(1, 1, 1), pad_mode=pad_mode,
                            norm_mode=norm_mode, act_mode=act_mode),
            # conv3d_norm_act(out_planes, out_planes, kernel_size=(3, 3, 3), padding=(1, 1, 1), pad_mode=pad_mode,
            #                 norm_mode=norm_mode)
        )
        self.conv1_maspp = MASPP(inplanes=out_planes, reduction=2, ASPP_3D=True)

        # self.conv2 = nn.Sequential(
        #     conv3d_norm_act(in_planes, out_planes, kernel_size=(1, 3, 3), padding=(0, 1, 1), pad_mode=pad_mode,
        #                     norm_mode=norm_mode, act_mode=act_mode),
        #     # conv3d_norm_act(out_planes, out_planes, kernel_size=(1, 3, 3), padding=(0, 1, 1), pad_mode=pad_mode,
        #     #                 norm_mode=norm_mode)
        # )
        # self.conv2_maspp = MASPP(inplanes=out_planes, reduction=3, ASPP_3D=False)
        #
        # self.fusion = conv3d_norm_act(out_planes, out_planes, kernel_size=(1, 1, 1), padding=(0, 0, 0), pad_mode=pad_mode,
        #                     norm_mode=norm_mode)

        self.projector = conv3d_norm_act(in_planes, out_planes, kernel_size=(1, 1, 1), padding=(0, 0, 0),
                                         norm_mode=norm_mode)
        self.act = get_layer_act(act_mode)[0]

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv1_maspp(y)
        #
        # y = self.fusion(y1+y2)
        if self.projection:
            y = y + self.projector(x)
        else:
            y = y + x
        y = self.act(y)
        return y



class residual_block_3d(nn.Module):
    """Residual Block 3D

    Args:
        in_planes (int): number of input channels.
        out_planes (int): number of output channels.
        projection (bool): projection of the input with a conv layer.
    """
    def __init__(self, in_planes, out_planes, projection=False, pad_mode='rep', norm_mode='bn', act_mode='elu'):
        super(residual_block_3d, self).__init__()
        self.projection = projection
        self.conv = nn.Sequential(
            conv3d_norm_act(in_planes, out_planes, kernel_size=(3,3,3), padding=(1,1,1), pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode),
            conv3d_norm_act(out_planes, out_planes, kernel_size=(3,3,3), padding=(1,1,1), pad_mode=pad_mode, norm_mode=norm_mode)
        )
        self.projector = conv3d_norm_act(in_planes, out_planes, kernel_size=(1,1,1), padding=(0,0,0), norm_mode=norm_mode)
        self.act = get_layer_act(act_mode)[0]
        
    def forward(self, x):
        y = self.conv(x)
        if self.projection:
            y = y + self.projector(x)
        else:
            y = y + x
        y = self.act(y)
        return y       

class bottleneck_dilated_2d(nn.Module):
    """Bottleneck Residual Block 2D with Dilated Convolution

    Args:
        in_planes (int): number of input channels.
        out_planes (int): number of output channels.
        projection (bool): projection of the input with a conv layer.
        dilate (int): dilation rate of conv filters.
    """
    def __init__(self, in_planes, out_planes, projection=False, dilate=2, pad_mode='rep', norm_mode='bn', act_mode='elu'):
        super(bottleneck_dilated_2d, self).__init__()
        self.projection = projection
        self.conv = nn.Sequential(
            conv2d_norm_act(in_planes, out_planes, kernel_size=(1, 1), padding=(0, 0), norm_mode=norm_mode, act_mode=act_mode),
            conv2d_norm_act(out_planes, out_planes, kernel_size=(3, 3), dilation=(dilate, dilate), padding=(dilate, dilate), pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode),
            conv2d_norm_act(out_planes, out_planes, kernel_size=(1, 1), padding=(0, 0), norm_mode=norm_mode)
        )
        self.projector = conv2d_norm_act(in_planes, out_planes, kernel_size=(1, 1), padding=(0, 0), norm_mode=norm_mode)
        self.act = get_layer_act(act_mode)[0]

    def forward(self, x):
        y = self.conv(x)
        if self.projection:
            y = y + self.projector(x)
        else:
            y = y + x
        y = self.act(y)
        return y

class bottleneck_dilated_3d(nn.Module):
    """Bottleneck Residual Block 3D with Dilated Convolution

    Args:
        in_planes (int): number of input channels.
        out_planes (int): number of output channels.
        projection (bool): projection of the input with a conv layer.
        dilate (int): dilation rate of conv filters.
    """
    def __init__(self, in_planes, out_planes, projection=False, dilate=2, pad_mode='rep', norm_mode='bn', act_mode='elu'):
        super(bottleneck_dilated_3d, self).__init__()
        self.projection = projection
        self.conv = nn.Sequential(
            conv3d_norm_act( in_planes, out_planes, kernel_size=(1,1,1), padding=(0,0,0), norm_mode=norm_mode, act_mode=act_mode),
            conv3d_norm_act(out_planes, out_planes, kernel_size=(3,3,3), 
                          dilation=(1, dilate, dilate), padding=(1, dilate, dilate), pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode),
            conv3d_norm_act(out_planes, out_planes, kernel_size=(1,1,1), padding=(0,0,0), norm_mode=norm_mode)
        )
        self.projector = conv3d_norm_act(in_planes, out_planes, kernel_size=(1,1,1), padding=(0,0,0), norm_mode=norm_mode)
        self.act = get_layer_act(act_mode)[0]

    def forward(self, x):
        y = self.conv(x)
        if self.projection:
            y = y + self.projector(x)
        else:
            y = y + x
        y = self.act(y)
        return y        
