import sys
import os
from optparse import OptionParser
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import cv2
from eval import eval_net
from unet import UNet
from utils import get_ids, split_ids, split_train_val, get_full_img_and_mask, batch, split_mask_image

def train_net(net,
              epochs=5,
              batch_size=1,
              lr=0.1,
              val_percent=0.05,
              save_cp=True,
              gpu=False,
              img_scale=(1024,1024),
            #"background","class1","class2","class3"
              class_mark = [0,1,2,3]
              ):

    dir_img = 'data/train/'
    dir_mask = 'data/train_masks/'
    dir_checkpoint = 'checkpoints/'
   

    ids = get_ids(dir_img)
    ids = split_ids(ids)

    # iddataset dictionary
    # {'train': dataset[:-n], 'val': dataset[-n:]}
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
               len(iddataset['val']), str(save_cp), str(gpu)))

    N_train = len(iddataset['train'])

    optimizer = optim.SGD(net.parameters(),
                          lr=lr,
                          momentum=0.9,
                          weight_decay=0.0005)

    criterion = nn.BCELoss()
    
    

    # reset the generators
    # train[(image1,mask1),(image2,mask2),...]
    train = get_full_img_and_mask(iddataset['train'], dir_img, dir_mask,'.png','_.png',img_scale)
    val = get_full_img_and_mask(iddataset['val'], dir_img, dir_mask,'.png','_.png',img_scale)
 
        
    for epoch in range(epochs):
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
        epoch_loss = 0
        for i, b in enumerate(batch(train, batch_size)):
            imgs = np.array([i[0] for i in b]).astype(np.float32)
            #imgs.shape = (batch_size, channels(3),height,width)
            true_masks = np.array([i[1] for i in b])
            #true_masks.shape = (batch_size,height,width)
            #split a n_class mask into n masks
            true_masks = split_mask_image(true_masks,class_mark)
            #true_masks.shape = (batch_size,n,height,width)
            imgs = torch.from_numpy(imgs)
            true_masks = torch.from_numpy(true_masks)
            
            if gpu:
                imgs = imgs.cuda()
                true_masks = true_masks.cuda()
            
            masks_pred = net(imgs)
            #masks_pred.shape = (batch size, n, height,width)
            masks_probs = F.sigmoid(masks_pred)
            masks_probs_flat = masks_probs.view(-1)

            true_masks_flat = true_masks.view(-1)
            loss = criterion(masks_probs_flat, true_masks_flat)
            epoch_loss += loss.item()
            print('loss: {0:.6f} '.format(loss.item()))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Epoch finished ! Loss: {}'.format(epoch_loss))

        if 1:
            val_dice = eval_net(net, val, gpu)
            print('Validation Dice Coeff: {}'.format(val_dice))

        if save_cp:
            torch.save(net.state_dict(),
                       dir_checkpoint + 'CP{}.pth'.format(epoch + 1))
            print('Checkpoint {} saved !'.format(epoch + 1))



def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=50, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batchsize', default= 2,
                      type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.001,
                      type='float', help='learning rate')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default= False, help='use cuda')
    parser.add_option('-c', '--load', dest='load',
                      default=False, help='load file model')
    parser.add_option('-s', '--scale', dest='scale', 
                      default=(1024,1024), help='target size')
    parser.add_option('-a', '--class', dest='class_mark',
                      default=[0,1,2,3], help='intensity in mask image')
    (options, args) = parser.parse_args()
    return options

if __name__ == '__main__':
    args = get_args()

    net = UNet(n_channels=3, n_classes=3)#@

    if args.load:
        net.load_state_dict(torch.load(args.load))
        print('Model loaded from {}'.format(args.load))

    if args.gpu:
        net.cuda()
        # cudnn.benchmark = True # faster convolutions, but more memory

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  gpu=args.gpu,
                  img_scale=args.scale
                  class_mark=args.class_mark)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
