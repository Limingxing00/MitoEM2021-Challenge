import os,sys

import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from ..block import *
from ..utils import *

class BilinearUp(nn.Module):
	def __init__(self, in_channels, out_channels, factor=(1,2,2)):
		super(BilinearUp, self).__init__()
		assert in_channels==out_channels
		self.groups = in_channels
		self.factor = factor
		self.kernel_size = [(2 * f) - (f % 2) for f in self.factor]
		self.padding = [int(math.ceil((f - 1) / 2.0)) for f in factor]
		self.init_weights()

	def init_weights(self):
		weight = torch.Tensor(self.groups, 1, *self.kernel_size)
		width = weight.size(-1)
		hight = weight.size(-2)
		assert width==hight
		f = float(math.ceil(width / 2.0))
		c = float(width - 1) / (2.0 * f)
		for w in range(width):
			for h in range(hight):
				weight[...,h,w] = (1 - abs(w/f - c)) * (1 - abs(h/f - c))
		self.register_buffer('weight', weight) # fixed

	def forward(self, x):
		return F.conv_transpose3d(x, self.weight, stride=self.factor, padding=self.padding, groups=self.groups)


class unet_residual_3d(nn.Module):
    """Lightweight 3D U-net with residual blocks (based on [Lee2017]_ with modifications).

    .. [Lee2017] Lee, Kisuk, Jonathan Zung, Peter Li, Viren Jain, and 
        H. Sebastian Seung. "Superhuman accuracy on the SNEMI3D connectomics 
        challenge." arXiv preprint arXiv:1706.00120, 2017.
        
    Args:
        in_channel (int): number of input channels.
        out_channel (int): number of output channels.
        filters (list): number of filters at each u-net stage.
    """
    def __init__(self, in_channel=1, out_channel=3, filters=[28, 36, 48, 64, 80], pad_mode='rep', norm_mode='bn', act_mode='elu', 
                 do_embedding=True, head_depth=1, output_act='sigmoid'):
        super().__init__()

        self.depth = len(filters)-2
        self.do_embedding = do_embedding
        self.output_act = output_act # activation function for the output layer

        # encoding path
        if self.do_embedding: 
            num_out = filters[1]
            self.downE = nn.Sequential(
                # anisotropic embedding
                conv3d_norm_act(in_planes=in_channel, out_planes=filters[0], 
                              kernel_size=(1,5,5), stride=1, padding=(0,2,2), pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode),
                # 2d residual module
                conv3d_norm_act(in_planes=filters[0], out_planes=filters[0], 
                              kernel_size=(1,3,3), stride=1, padding=(0,1,1), pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode),
                residual_block_3d(filters[0], filters[0], projection=False, pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode)
            )
        else:
            filters[0] = in_channel
            num_out = out_channel
        
        self.downC = nn.ModuleList(
            [nn.Sequential(
            conv3d_norm_act(in_planes=filters[x], out_planes=filters[x+1], 
                          kernel_size=(1,3,3), stride=1, padding=(0,1,1), pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode),
            residual_block_3d(filters[x+1], filters[x+1], projection=False, pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode)
            ) for x in range(self.depth)])

        # pooling downsample
        # self.downS = nn.ModuleList([nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2)) for x in range(self.depth+1)])
        self.downS = nn.ModuleList(  # 用conv3d 下采样
            [conv3d_norm_act(in_planes=filters[x], out_planes=filters[x], kernel_size=(1,3,3), stride=(1, 2, 2), padding=(0,1,1), pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode)
            for x in range(self.depth+1)])

        # center block
        self.center = nn.Sequential(conv3d_norm_act(in_planes=filters[-2], out_planes=filters[-1], 
                          kernel_size=(1,3,3), stride=1, padding=(0,1,1), pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode),
            residual_block_3d(filters[-1], filters[-1], projection=True)
        )
        self.middle = nn.ModuleList(
            [nn.Sequential(
                conv3d_norm_act(in_planes=filters[x], out_planes=filters[x],
                          kernel_size=(1,3,3), stride=1, padding=(0,1,1), pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode)
            ) for x in range(self.depth+1)])
            
        self.upC = nn.ModuleList(
            [nn.Sequential(
                conv3d_norm_act(in_planes=filters[x+1], out_planes=filters[x+1], 
                          kernel_size=(1,3,3), stride=1, padding=(0,1,1), pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode),
                residual_block_3d(filters[x+1], filters[x+1], projection=False)
            ) for x in range(self.depth)])

        if self.do_embedding: 
            # decoding path
            self.upE = nn.Sequential(
                conv3d_norm_act(in_planes=filters[0], out_planes=filters[0], 
                              kernel_size=(1,3,3), stride=1, padding=(0,1,1), pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode),
                residual_block_3d(filters[0], filters[0], projection=False, pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode),
                conv3d_norm_act(in_planes=filters[0], out_planes=out_channel, 
                              kernel_size=(1,5,5), stride=1, padding=(0,2,2), pad_mode=pad_mode, norm_mode=norm_mode)
            )
            # conv + upsample
            self.upS = nn.ModuleList([nn.Sequential(
                            conv3d_norm_act(filters[x+1], filters[x], kernel_size=(1,1,1), padding=0, norm_mode=norm_mode, act_mode=act_mode),
                            BilinearUp(filters[x], filters[x], factor=(1,2,2))) for x in range(self.depth+1)])
        else:
            # new
            head_pred = [residual_block_3d(filters[1], filters[1], projection=False)
                                    for x in range(head_depth-1)] + \
                              [conv3d_norm_act(filters[1], out_channel, kernel_size=(1,1,1), padding=0, norm_mode=norm_mode)]
            self.upS = nn.ModuleList( [nn.Sequential(*head_pred)] + \
                                 [nn.Sequential(
                        conv3d_norm_act(filters[x+1], filters[x], kernel_size=(1,1,1), padding=0, norm_mode=norm_mode, act_mode=act_mode),
                                     BilinearUp(filters[x], filters[x], factor=(1, 2, 2))) for x in range(1,self.depth+1)])
            """
            # old
            self.upS = nn.ModuleList( [conv3d_norm_act(filters[1], out_channel, kernel_size=(1,1,1), padding=0, norm_mode=norm_mode)] + \
                                 [nn.Sequential(
                        conv3d_norm_act(filters[x+1], filters[x], kernel_size=(1,1,1), padding=0, norm_mode=norm_mode, act_mode=act_mode),
                        nn.Upsample(scale_factor=(1,2,2), mode='trilinear', align_corners=False)) for x in range(1,self.depth+1)])
            """


        #initialization
        ortho_init(self)

    def forward(self, x):
        # encoding path
        if self.do_embedding:
            z = self.downE(x) # torch.Size([4, 1, 32, 256, 256])
            x = self.downS[0](z) # downsample

        down_u = [None] * (self.depth)
        for i in range(self.depth):
            down_u[i] = self.downC[i](x)
            x = self.downS[i+1](down_u[i]) # downsample

        x = self.center(x)

        # z torch.Size([2, 28, 32, 256, 256]

        # down_u[0] [2, 36, 32, 128, 128]
        # down_u[1] [2, 48, 32, 64, 64]
        # down_u[3] [2, 64, 32, 32, 32]
        # x [16, 16]
        z = self.middle[0](z)
        layer = []

        # layer1 = self.middle[1](down_u[0])
        # layer2 = self.middle[2](down_u[1])
        # layer3 = self.middle[3](down_u[2])
        # layer4 = down_u[3]

        for i in range(len(down_u)-1):
            layer.append(self.middle[i+1](down_u[i]))
            # print(i)





        # decoding path
        for i in range(self.depth-1,-1,-1):
            x = down_u[i] + self.upS[i+1](x)
            x = self.upC[i](x)

        if self.do_embedding: 
            x = z + self.upS[0](x)
            x = self.upE(x)
        else:
            x = self.upS[0](x)

        x = get_functional_act(self.output_act)(x)
        return x