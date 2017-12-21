import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.nn as nn

from utils import *
# from myloss import DiceLoss
from eval import eval_net
from unet import UNet
from torch.autograd import Variable
from torch import optim
from optparse import OptionParser
import sys
import os
import numpy as np


class myLoss(nn.Module):
    """Custom loss function.
     """
    # def __init__(self, margin):
    def __init__(self):
        super(myLoss, self).__init__()
        # self.margin = margin

    def forward(self, a, b):
        # a_numpy = a.data.numpy()
        # b_numpy = b.data.numpy()
        # d = a_numpy - b_numpy
        # return np.sum(np.power(d, 2)) - np.sum(d)*np.sum(d)/2/np.prod(d.shape)
        # loss = nn.MSELoss(a, b) - 0.5*nn.L1Loss(a, b)*nn.L1Loss(a, b)
        # import pdb; pdb.set_trace()
        log_a  = torch.log(a)
        np_log_a = log_a.cpu().data.numpy()
        np_log_a = np.nan_to_num(np_log_a, 0)
        log_a = Variable(torch.from_numpy(np_log_a), requires_grad=True).cuda()
        log_b = torch.log(b)
        np_log_b = log_b.cpu().data.numpy()
        np_log_b = np.nan_to_num(np_log_b, 0)
        log_b = Variable(torch.from_numpy(np_log_b), requires_grad=True).cuda()

        d = log_a - log_b
        test1 =  torch.sum(d**2)
        test2 = torch.sum(d)**2
        print (test1, test2)
        lamda = 0.5
        loss = torch.sum(d**2)/(768000) - lamda*torch.sum(d)**2/(768000**2)
        # loss = torch.sum(d**2)/(768000) - torch.sum(d)**2/(768000**2)
        # import pdb; pdb.set_trace()
        return loss

def train_net(net, epochs=5, batch_size=10, lr=0.1, val_percent=0.05,
              cp=True, gpu=False, mask_type="depth"):
    prefix = "/data/chc631/project/"
    # prefix = ""
    dir_img = prefix + 'data/train/'
    if mask_type == "depth":
        dir_mask = prefix + "data/train_masks_depth_map/"
    else:
        dir_mask = prefix + 'data/train_masks/'
    dir_checkpoint = 'checkpoints/'

    ids = get_ids(dir_img)
    ids = split_ids(ids)

    iddataset = split_train_val(ids, val_percent)

    print('''
    Starting training:
        Epochs: {}
        Batch size: {}
        Learning rate: {}
        Training size: {}
        Validation size: {}
        Checkpoints: {}
        CUDA: {}
    '''.format(epochs, batch_size, lr, len(iddataset['train']),
               len(iddataset['val']), str(cp), str(gpu)))

    N_train = len(iddataset['train'])

    train = get_imgs_and_masks(iddataset['train'], dir_img, dir_mask)
    val = get_imgs_and_masks(iddataset['val'], dir_img, dir_mask)

    optimizer = optim.SGD(net.parameters(),
                          lr=lr, momentum=0.9, weight_decay=0.0005)
    # criterion = nn.BCELoss()
    criterion = myLoss()
    # criterion = nn.SmoothL1Loss()
    # criterion = nn.MSELoss()
    # criterion = nn.L1Loss()

    for epoch in range(epochs):
        print('Starting epoch {}/{}.'.format(epoch+1, epochs))
        # train = get_imgs_and_masks(iddataset['train'], dir_img, dir_mask)
        # val = get_imgs_and_masks(iddataset['val'], dir_img, dir_mask)
        # import pdb; pdb.set_trace()

        epoch_loss = 0

        # if 0:
        #     val_dice = eval_net(net, val, gpu)
        #     print('Validation Dice Coeff: {}'.format(val_dice))

        for i, b in enumerate(batch(train, batch_size)):
            X = np.array([i[0] for i in b])
            y = np.array([i[1] for i in b])

            X = torch.FloatTensor(X)
            y = torch.FloatTensor(y)
            y = y.unsqueeze(0) # manually create a channel dimension for conv2d
            y = y.transpose(0, 1)

            if gpu:
                X = Variable(X).cuda()
                y = Variable(y).cuda()
            else:
                X = Variable(X)
                y = Variable(y)

            # import pdb; pdb.set_trace()
            y_pred = net(X)
            # probs = F.sigmoid(y_pred)
            # probs_flat = probs.view(-1)
            y_pred_flat = y_pred.view(-1)

            # y = Variable(y)
            conv_mat = Variable(torch.ones(1,1,2,2)).cuda()
            y = F.conv2d(y, conv_mat, stride=2)
            y = torch.squeeze(y)

            y_flat = y.view(-1)

            # import pdb; pdb.set_trace()
            loss = criterion(y_pred_flat, y_flat.float())
            # loss = criterion(y_pred, y.long())
            epoch_loss += loss.data[0]

            print('{0:.4f} --- loss: {1:.6f}'.format(i*batch_size/N_train,
                                                     loss.data[0]))

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

        print('Epoch finished ! Loss: {}'.format(epoch_loss/i))

        if cp:
            torch.save(net.state_dict(),
                       dir_checkpoint + 'CP{}.pth'.format(epoch+1))

            print('Checkpoint {} saved !'.format(epoch+1))


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=5, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batchsize', default=10,
                      type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.1,
                      type='float', help='learning rate')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=False, help='use cuda')
    parser.add_option('-c', '--load', dest='load',
                      default=False, help='load file model')

    (options, args) = parser.parse_args()

    net = UNet(3, 1)

    if options.load:
        net.load_state_dict(torch.load(options.load))
        print('Model loaded from {}'.format(options.load))

    if options.gpu:
        net.cuda()
        cudnn.benchmark = True

    try:
        train_net(net, options.epochs, options.batchsize, options.lr,
                  gpu=options.gpu)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
