#!/usr/bin/python
# full assembly of the sub-parts to form the complete net

import torch
import torch.nn as nn
import torch.nn.functional as F

# python 3 confusing imports :(
from .unet_parts import *


class UNet4(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet4, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        # self.down1_5 = down(128, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 256)

        # self.down4 = down(512, 512)
        # self.up1 = up(1024, 256)
        # self.up2_5 = up(256, 128)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.up4_2 = up(192, 64)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        # import pdb; pdb.set_trace()
        x1 = self.inc(x)
        # import pdb; pdb.set_trace()
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        # x5 = self.down4(x4)
        # x = self.up1(x5, x4)
        x = self.up2(x4, x3)
        # x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        # x = self.down1_5(x2)
        # import pdb; pdb.set_trace()
        # x = self.up4_2(x2, x1)
        # import pdb; pdb.set_trace()
        x = self.outc(x)
        return x
