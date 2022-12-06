import torch
from unet import UNet as _UNet

def unet_carvana(pretrained=False, scale=0.5):
    """
    UNet model trained on the Carvana dataset ( https://www.kaggle.com/c/carvana-image-masking-challenge/data ).
    Set the scale to 0.5 (50%) when predicting.
    """
    net = _UNet(n_channels=3, n_classes=2, bilinear=False)
    if pretrained:
        if scale == 0.5:
            checkpoint = 'https://github.com/milesial/Pytorch-UNet/releases/download/v4.0/unet_carvana_scale0.5_epoch5.pth'
        else:
            raise RuntimeError('Only 0.5 scale is available')
        state_dict = torch.hub.load_state_dict_from_url(checkpoint, progress=True)
        del state_dict['mask_values']
        net.load_state_dict(state_dict)

    return net
