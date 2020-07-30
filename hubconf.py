import torch
from unet import UNet as _UNet

def unet_carvana(pretrained=False):
    """
    UNet model trained on the Carvana dataset ( https://www.kaggle.com/c/carvana-image-masking-challenge/data ).
    Set the scale to 1 (100%) when predicting.
    """
    net = _UNet(n_channels=3, n_classes=1, bilinear=True)
    if pretrained:
        checkpoint = 'https://github.com/milesial/Pytorch-UNet/releases/download/v1.0/unet_carvana_scale1_epoch5.pth'
        net.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint, progress=True))

    return net

