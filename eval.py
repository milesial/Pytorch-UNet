import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from scaleInvarLoss import scaleInvarLoss


def eval_net(net, dataset, gpu=False, half_scale=True):
    epoch_loss = 0
    count = 0
    for i, b in enumerate(dataset):
        X = b[0]
        y = b[1]

        X = torch.FloatTensor(X).unsqueeze(0)
        y = torch.FloatTensor(y).unsqueeze(0)

        if gpu:
            X = Variable(X, volatile=True).cuda()
            y = Variable(y, volatile=True).cuda()
        else:
            X = Variable(X, volatile=True)
            y = Variable(y, volatile=True)

        y_pred = net(X)
        y_pred_flat = y_pred.view(-1)
        if half_scale:
            y = y.unsqueeze(0)
            conv_mat = Variable(torch.ones(1,1,2,2)).cuda()
            y = F.conv2d(y, conv_mat, stride=2)
            y = torch.squeeze(y)

        y_flat = y.view(-1)
        criterion = scaleInvarLoss()
        loss = criterion(y_pred_flat, y_flat.float())
        epoch_loss += loss.data[0]
        count+=1

    return epoch_loss / count
