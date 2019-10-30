import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from eval import eval_net
from unet import UNet
from utils import get_ids, split_train_val, get_imgs_and_masks, batch

dir_img = 'data/imgs/'
dir_mask = 'data/masks/'
dir_checkpoint = 'checkpoints/'


def train_net(net,
              device,
              epochs=5,
              batch_size=1,
              lr=0.1,
              val_percent=0.15,
              save_cp=True,
              img_scale=0.5):
    ids = get_ids(dir_img)

    iddataset = split_train_val(ids, val_percent)

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {len(iddataset["train"])}
        Validation size: {len(iddataset["val"])}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')

    n_train = len(iddataset['train'])
    n_val = len(iddataset['val'])
    optimizer = optim.Adam(net.parameters(), lr=lr)
    if net.n_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        net.train()

        # reset the generators
        train = get_imgs_and_masks(iddataset['train'], dir_img, dir_mask, img_scale)
        val = get_imgs_and_masks(iddataset['val'], dir_img, dir_mask, img_scale)

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for i, b in enumerate(batch(train, batch_size)):
                imgs = np.array([i[0] for i in b]).astype(np.float32)
                true_masks = np.array([i[1] for i in b])

                imgs = torch.from_numpy(imgs)
                true_masks = torch.from_numpy(true_masks)

                imgs = imgs.to(device=device)
                true_masks = true_masks.to(device=device)

                masks_pred = net(imgs)
                loss = criterion(masks_pred, true_masks)
                epoch_loss += loss.item()

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.update(batch_size)

        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')

        val_score = eval_net(net, val, device, n_val)
        if net.n_classes > 1:
            logging.info('Validation cross entropy: {}'.format(val_score))

        else:
            logging.info('Validation Dice Coeff: {}'.format(val_score))


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=5,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.1,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.5,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=15.0,
                        help='Percent of the data that is used as validation (0-100)')

    return parser.parse_args()


def pretrain_checks():
    imgs = [f for f in os.listdir(dir_img) if not f.startswith('.')]
    masks = [f for f in os.listdir(dir_mask) if not f.startswith('.')]
    if len(imgs) != len(masks):
        logging.warning(f'The number of images and masks do not match ! '
                        f'{len(imgs)} images and {len(masks)} masks detected in the data folder.')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    pretrain_checks()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N
    net = UNet(n_channels=3, n_classes=1)
    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Dilated conv"} upscaling')

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    # faster convolutions, but more memory
    # cudnn.benchmark = True

try:
    train_net(net=net,
              epochs=args.epochs,
              batch_size=args.batchsize,
              lr=args.lr,
              device=device,
              img_scale=args.scale,
              val_percent=args.val / 100)
except KeyboardInterrupt:
    torch.save(net.state_dict(), 'INTERRUPTED.pth')
    logging.info('Saved interrupt')
    try:
        sys.exit(0)
    except SystemExit:
        os._exit(0)
