
#
# myloss.py : implementation of the Dice coeff and the associated loss
#

import torch
import torch.nn as nn

from torch.nn.modules.loss import _Loss
from torch.autograd import Function, Variable

class scaleInvarLoss(nn.Module):
    """Custom loss function.
     """
    def __init__(self):
        super(scaleInvarLoss, self).__init__()

    def forward(self, a, b, eps = 5, lamda = 0.5):

        """
        a: prediction from Net()
        b: target
        The problem right now is log(a) might return NaN, alot, this version of forward  >>\add a constant to negate the issue.
        """

        log_a  = torch.log(a + eps)
        log_b = torch.log(b + eps)
        # Tried set Nan to 0 spits an error
        # log_a[log_a != log_a] = 0

        d = log_a - log_b
        size = d.size()[0]
        loss = torch.sum(d**2)/(size) - lamda*torch.sum(d)**2/(size**2)
        return loss

