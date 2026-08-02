""" Full assembly of the parts to form the complete network """

import torch.utils.checkpoint as cp

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.use_checkpointing = False

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        def run(module, *inputs):
            if self.use_checkpointing:
                return cp.checkpoint(module, *inputs, use_reentrant=False)
            return module(*inputs)

        x1 = run(self.inc, x)
        x2 = run(self.down1, x1)
        x3 = run(self.down2, x2)
        x4 = run(self.down3, x3)
        x5 = run(self.down4, x4)
        x = run(self.up1, x5, x4)
        x = run(self.up2, x, x3)
        x = run(self.up3, x, x2)
        x = run(self.up4, x, x1)
        logits = self.outc(x)
        return logits