""" Full assembly of the parts to form the complete network """

from .unet_parts import *
import pytorch_lightning as pl


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class CustomUNet(pl.LightningModule):
    def __init__(
            self,
            num_channels: int = 1,
            num_classes: int = 1,
            filters: int = 16,
            num_layers: int = 4,
            bilinear: bool = False,
            learning_rate = 0.001
    ):
        super().__init__()
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.bilinear = bilinear
        self.num_layers = num_layers
        self.filters = filters
        self.learning_rate = learning_rate

        self.inc = DoubleConv(self.num_channels, self.filters)
        self.outc = OutConv(self.filters, self.num_classes)
        self.output_activation = nn.Sigmoid()

        self.down_list = nn.ModuleList()
        self.up_list = nn.ModuleList()

        factor = 2
        in_channels = self.filters
        for i in range(self.num_layers):
            down = Down(in_channels, in_channels * factor)
            up = Up(in_channels * factor, in_channels, self.bilinear)

            self.down_list.append(down)
            self.up_list.append(up)
            in_channels *= factor
        self.up_list = self.up_list[::-1]

    def conv(self, x):
        emb_list = []
        x = self.inc(x)
        for down in self.down_list:
            emb_list.append(x)
            x = down(x)
        emb_list = emb_list[::-1]
        return x, emb_list

    def deconv(self, x, emb_list):
        for up, emb in zip(self.up_list, emb_list):
            x = up(x, emb)
        logits = self.outc(x)
        return logits

    def forward(self, x):
        emb, emb_list = self.conv(x)
        logits = self.deconv(emb, emb_list)
        pred = self.output_activation(logits)
        return pred

    def training_step(self, batch, batch_idx):
        x, y = batch
        emb, emb_list = self.conv(x)
        logits = self.deconv(emb, emb_list)
        y_hat = self.output_activation(logits)
        loss = F.mse_loss(y_hat, y)
        # self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        tensorboard_logs = {
            'train_loss': loss,
        }

        result = {
            'loss': loss,
            'log': tensorboard_logs,
        }
        return result

    def validation_step(self, batch, batch_idx):
        x, y = batch
        emb, emb_list = self.conv(x)
        logits = self.deconv(emb, emb_list)
        y_hat = self.output_activation(logits)
        loss = F.mse_loss(y_hat, y)
        # self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        tensorboard_logs = {
            'val_loss': loss,
        }

        result = {
            'val_loss': loss,
            'log': tensorboard_logs,
        }
        return result

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        lr_scheduler = {'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                                        optimizer,
                                        verbose=True,
                                    ),
                        'name': 'learning_rate',
                        'interval': 'step',
                        'frequency': 1}

        return [optimizer], [lr_scheduler]
