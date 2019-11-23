import torch
import torch.nn.functional as F
from tqdm import tqdm

from dice_loss import dice_coeff


def eval_net(net, loader, device, n_val):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    tot = 0

    for i, b in tqdm(enumerate(loader), desc='Validation round', unit='img'):
        imgs = b['image']
        true_masks = b['mask']

        imgs = imgs.to(device=device, dtype=torch.float32)
        true_masks = true_masks.to(device=device, dtype=torch.float32)

        mask_pred = net(imgs)

        for true_mask in true_masks:
            mask_pred = (mask_pred > 0.5).float()
            if net.n_classes > 1:
                tot += F.cross_entropy(mask_pred.unsqueeze(dim=0), true_mask.unsqueeze(dim=0)).item()
            else:
                tot += dice_coeff(mask_pred, true_mask.squeeze(dim=1)).item()

    return tot / n_val
