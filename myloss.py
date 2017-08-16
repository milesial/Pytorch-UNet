import torch
from torch.nn.modules.loss import _Loss
from torch.autograd import Function
import torch.nn.functional as F

class DiceCoeff(Function):

    def forward(ctx, input, target):
        ctx.save_for_backward(input, target)
        ctx.inter = torch.dot(input, target) + 0.0001
        ctx.union = torch.sum(input) + torch.sum(target) + 0.0001

        t = 2*ctx.inter.float()/ctx.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(ctx, grad_output):

        input, target = ctx.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * ctx.union + ctx.inter) \
                         / ctx.union * ctx.union
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target

def dice_coeff(input, target):
    return DiceCoeff().forward(input, target)

class DiceLoss(_Loss):
    def forward(self, input, target):
        return 1 - dice_coeff(F.sigmoid(input), target)
