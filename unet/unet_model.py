from .unet_parts import *
from .USM import USM


class UNet(nn.Module):
    ''' This Object defines the architecture of U-Net. '''

    def __init__(self, n_channels, n_classes, with_USM=False, batch_len=10):
        super(UNet, self).__init__()

        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        # TODO: Use or not the bilinear option
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)

        self.with_USM = with_USM

        # USM
        if(self.with_USM):
            self.usm = USM(in_channels=n_channels,  kernel_size=3,
                           sigma=0.005,  batch_len=batch_len)

    def forward(self, x):
        if(self.with_USM):
            x_p = self.usm(x)
            x1 = self.inc(x_p)
        else:
            x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x
