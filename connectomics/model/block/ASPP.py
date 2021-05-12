# -*- coding: utf-8 -*-
# @Time    : 2021/1/7 21:54
# @Author  : Mingxing Li
# @FileName: ASPP.py
# @Software: PyCharm

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb


# from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv3d(inplanes, planes, kernel_size=kernel_size,
                                     stride=1, padding=padding, dilation=dilation, bias=False)
        # self.bn = BatchNorm(planes)
        self.relu = nn.ELU(inplace=True)

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        # x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)
            # elif isinstance(m, SynchronizedBatchNorm2d):
            # m.weight.data.fill_(1)
            # m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class MASPP(nn.Module):
    def __init__(self, inplanes, reduction=3, dilations=[1, 2, 4, 6], ASPP_3D=True):
        super(MASPP, self).__init__()
        # if output_stride == 16:
        #     dilations = [1, 6, 12, 18]
        # elif output_stride == 8:
        #     dilations = [1, 12, 24, 36]
        # elif output_stride == 4:
        #     dilations = [1, 2, 4, 6]
        # else:
        #     raise NotImplementedError


        # self.conv0 = nn.Conv3d(inplanes, inplanes, 1, bias=False)
        # self.bn1 = BatchNorm(inplanes)

        middle = int(inplanes//reduction)

        if ASPP_3D == True:
            self.aspp1 = _ASPPModule(inplanes, middle, kernel_size=3, padding=dilations[0], dilation=dilations[0])
            self.aspp2 = _ASPPModule(inplanes, middle, kernel_size=3, padding=dilations[1], dilation=dilations[1])
            self.aspp3 = _ASPPModule(inplanes, middle, kernel_size=3, padding=dilations[2], dilation=dilations[2])
            self.aspp4 = _ASPPModule(inplanes, middle, kernel_size=(1, 3, 3), padding=(0, dilations[3], dilations[3]),
                                     dilation=(1, dilations[3], dilations[3]))
        else:
            # 2d
            self.aspp1 = _ASPPModule(inplanes, middle, kernel_size=(1,3,3), padding=(0, dilations[0], dilations[0]),
                                     dilation=(1, dilations[0], dilations[0]))
            self.aspp2 = _ASPPModule(inplanes, middle, kernel_size=(1,3,3), padding=(0, dilations[1], dilations[1]),
                                     dilation=(1, dilations[1], dilations[1]))
            self.aspp3 = _ASPPModule(inplanes, middle, kernel_size=(1,3,3), padding=(0, dilations[2], dilations[2]),
                                     dilation=(1, dilations[2], dilations[2]))



        # self.aspp4 = _ASPPModule(inplanes, planes, 3, padding=dilations[3], dilation=dilations[3], BatchNorm=BatchNorm)

        # self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
        #                                      nn.Conv3d(inplanes, planes, 1, stride=1, bias=False),
        #                                      BatchNorm(planes),
        #                                      nn.ReLU(inplace=True))

        self.conv1 = nn.Conv3d(4 * middle, inplanes, 1)
        # self.bn1 = BatchNorm(inplanes)
        self.relu = nn.ELU(inplace=True)
        # @ self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        # pdb.set_trace()
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        # x5 = self.global_avg_pool(x)
        # x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv1(x)
        # x = self.bn1(x)
        x = self.relu(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            # elif isinstance(m, SynchronizedBatchNorm2d):
            # m.weight.data.fill_(1)
            # m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()



if __name__=="__main__":

    # test 3d
    a = torch.rand(3, 10, 25, 25, 25)
    maspp_3d = MASPP(inplanes=10, reduction=3)
    b = maspp_3d(a)
    print(b.shape)

    # test 2d
    a = torch.rand(3, 10, 25, 25, 25)
    maspp_3d = MASPP(inplanes=10, reduction=3, ASPP_3D=False)
    b = maspp_3d(a)
    print(b.shape)