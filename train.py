import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.nn as nn

from utils import *
from scaleInvarLoss import scaleInvarLoss
from eval import eval_net
from unet import UNet
from unet import UNet4
from torch.autograd import Variable
from torch import optim
from optparse import OptionParser
import sys
import os
import numpy as np


def train_net(net, epochs=5, batch_size=10, lr=0.1, val_percent=0.05,
              cp=True, gpu=False, mask_type="depth", half_scale=True):
    prefix = "/data/chc631/project/"
    dir_img = prefix + 'data/train/'
    # use depth map as target
    if mask_type == "depth":
        dir_mask = prefix + "data/train_masks_depth_map/"
    # use color map as target
    else:
        dir_mask = prefix + 'data/train_masks/'
    dir_checkpoint = "/data/chc631/project/data/checkpoints/" + options.dir
    if not os.path.exists(dir_checkpoint):
        os.makedirs(dir_checkpoint)

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
    # if half_scale:
    #     train = get_imgs_and_masks(iddataset['train'], dir_img, dir_mask, scale=0.5)
    #     val = get_imgs_and_masks(iddataset['val'], dir_img, dir_mask, scale=0.5)
    # else:
    #     train = get_imgs_and_masks(iddataset['train'], dir_img, dir_mask, scale=1)
    #     val = get_imgs_and_masks(iddataset['val'], dir_img, dir_mask, scale=1)

    optimizer = optim.SGD(net.parameters(),
                          lr=lr, momentum=0.9, weight_decay=0.0005)
    criterion = scaleInvarLoss()

    for epoch in range(epochs):
        net.train()
        print('Starting epoch {}/{}.'.format(epoch+1, epochs))
        epoch_loss = 0


        if half_scale:
            print ("half_scale")
            train = get_imgs_and_masks(iddataset['train'], dir_img, dir_mask, scale=0.5)
            val = get_imgs_and_masks(iddataset['val'], dir_img, dir_mask, scale=0.5)
        else:
            train = get_imgs_and_masks(iddataset['train'], dir_img, dir_mask, scale=1)
            val = get_imgs_and_masks(iddataset['val'], dir_img, dir_mask, scale=1)
        # train = get_imgs_and_masks(iddataset['train'], dir_img, dir_mask)
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

            y_pred = net(X)
            # probs = F.sigmoid(y_pred)
            # probs_flat = probs.view(-1)
            y_pred_flat = y_pred.view(-1)

            if half_scale:
                conv_mat = Variable(torch.ones(1,1,2,2)).cuda()
                y = F.conv2d(y, conv_mat, stride=2)
                y = torch.squeeze(y)

            y_flat = y.view(-1)

            loss = criterion(y_pred_flat, y_flat.float())
            epoch_loss += loss.data[0]

            print('{0:.4f} --- loss: {1:.6f}'.format(i*batch_size/N_train,
                                                     loss.data[0]))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Epoch finished ! Loss: {}'.format(epoch_loss/i))

        if cp:
            torch.save(net.state_dict(),
                       dir_checkpoint +"/" +'CP{}.pth'.format(epoch+1))

            print('Checkpoint {} saved !'.format(epoch+1))
            val_err = eval_net(net, val, gpu, half_scale)
            print('Validation Error: {}'.format(val_err))
            with open (dir_checkpoint+"/ValidationError.txt", 'a') as outfile:
                outfile.write(str(val_err)+ '\n')
            with open (dir_checkpoint+"/TrainingError.txt", 'a') as outfile:
                outfile.write(str(val_err)+ '\n')



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
    parser.add_option('--full', dest='full',
                      default=False, help='use full image')
    parser.add_option('--dir', dest='dir',
                      default='checkpoints/', type="string", help='saved model directory')

    (options, args) = parser.parse_args()

    net = UNet4(3, 1)

    if options.load:
        net.load_state_dict(torch.load(options.load))
        print('Model loaded from {}'.format(options.load))

    if options.gpu:
        net.cuda()
        cudnn.benchmark = True

    try:
        train_net(net, options.epochs, options.batchsize, options.lr,
                  gpu=options.gpu, half_scale = not options.full)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
