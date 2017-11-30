import torch
from myloss import dice_coeff
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn.functional as F

from utils import dense_crf, plot_img_mask


def eval_net(net, dataset, gpu=False):
    tot = 0
    for i, b in enumerate(dataset):
        X = b[0]
        y = b[1]

        X = torch.FloatTensor(X).unsqueeze(0)
        y = torch.ByteTensor(y).unsqueeze(0)

        if gpu:
            X = Variable(X, volatile=True).cuda()
            y = Variable(y, volatile=True).cuda()
        else:
            X = Variable(X, volatile=True)
            y = Variable(y, volatile=True)

        y_pred = net(X)

        y_pred = (F.sigmoid(y_pred) > 0.6).float()
        # y_pred = F.sigmoid(y_pred).float()

        dice = dice_coeff(y_pred, y.float()).data[0]
        tot += dice

        if 0:
            X = X.data.squeeze(0).cpu().numpy()
            X = np.transpose(X, axes=[1, 2, 0])
            y = y.data.squeeze(0).cpu().numpy()
            y_pred = y_pred.data.squeeze(0).squeeze(0).cpu().numpy()
            print(y_pred.shape)

            fig = plt.figure()
            ax1 = fig.add_subplot(1, 4, 1)
            ax1.imshow(X)
            ax2 = fig.add_subplot(1, 4, 2)
            ax2.imshow(y)
            ax3 = fig.add_subplot(1, 4, 3)
            ax3.imshow((y_pred > 0.5))

            Q = dense_crf(((X*255).round()).astype(np.uint8), y_pred)
            ax4 = fig.add_subplot(1, 4, 4)
            print(Q)
            ax4.imshow(Q > 0.5)
            plt.show()
    return tot / i
