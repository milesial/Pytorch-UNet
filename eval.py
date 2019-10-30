import torch
import torch.nn.functional as F
from tqdm import tqdm

from dice_loss import dice_coeff


def eval_net(net, dataset, device, n_val):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    tot = 0

    for i, b in tqdm(enumerate(dataset), total=n_val, desc='Validation round', unit='img'):
        img = b[0]
        true_mask = b[1]

        img = torch.from_numpy(img).unsqueeze(0)
        true_mask = torch.from_numpy(true_mask).unsqueeze(0)

        img = img.to(device=device)
        true_mask = true_mask.to(device=device)

        mask_pred = net(img).squeeze(dim=0)

        mask_pred = (mask_pred > 0.5).float()
        if net.n_classes > 1:
            tot += F.cross_entropy(mask_pred.unsqueeze(dim=0), true_mask.unsqueeze(dim=0)).item()
        else:
            tot += dice_coeff(mask_pred, true_mask.squeeze(dim=1)).item()

    return tot / n_val
