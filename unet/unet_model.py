""" Full assembly of the parts to form the complete network """

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, upscaling_mode='upsample'):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.upscaling_mode = upscaling_mode

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 1 if upscaling_mode == 'transpose' else 2
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, upscaling_mode)
        self.up2 = Up(512, 256 // factor, upscaling_mode)
        self.up3 = Up(256, 128 // factor, upscaling_mode)
        self.up4 = Up(128, 64, upscaling_mode)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2, indices2 = self.down1(x1)
        x3, indices3 = self.down2(x2)
        x4, indices4 = self.down3(x3)
        x5, indices5 = self.down4(x4)
        x = self.up1(x5, x4, indices5)
        x = self.up2(x, x3, indices4)
        x = self.up3(x, x2, indices3)
        x = self.up4(x, x1, indices2)
        logits = self.outc(x)
        return logits
