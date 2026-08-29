""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),  #对于Conv，stride默认等于1；对于Pooling，stride默认等于kernel_size。
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),                #对于Pooling，stride默认等于kernel_size。
            DoubleConv(in_channels, out_channels)  #Down默认mid_channels=out_channels，是在优先保证特征提取能力（贴合原论文结构），不是优化计算量。
        )                                          #它在复现原始U-Net论文的编码器（收缩路径）设计：论文里下采样阶段的每个"双卷积块"，两次卷积都工作在新的（更深的）通道数上——即先把通道数一次性提升到out_channels，然后在这个更高维度上再卷一次，用来在每次下采样后立刻获得更丰富的特征表达能力，而不是让特征提取"延迟"到第二层卷积才达到目标深度。

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  #Upsample只做上采样，不改变通道数。
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)  #bilinear分支的mid_channels=in_channels//2，是在弥补bilinear分支缺少可学习降维这一结构性差异（对比ConvTranspose2d）。
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)  #既做上采样，也把通道数减半（in_channels → in_channels // 2）。
            self.conv = DoubleConv(in_channels, out_channels)

        '''
        因为Upsample是无参数的插值，不像ConvTranspose2d那样能在上采样过程中学习着把1024通道压缩到512。
        为了让bilinear分支的DoubleConv获得和转置卷积分支相近的"表达能力"/参数量，作者特意让bilinear分支的第一层卷积也先降到in_channels//2（1024→512），而不是直接一步降到out_channels（1024→256）。
        这样两条分支在语义上大致对齐：都是先降到中间维度、再降到目标维度，只是"第一次降维"这一步，转置卷积分支是在self.up里做的，bilinear分支是挪到self.conv内部做的。
        '''

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
