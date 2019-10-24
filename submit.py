""" Submit code specific to the kaggle challenge"""

import os

import torch
from PIL import Image

from predict import predict_img
from unet import UNet
from utils import rle_encode


def submit(net, gpu=False):
    """Used for Kaggle submission: predicts and encode all test images"""
    dir = 'data/test/'

    N = len(list(os.listdir(dir)))
    with open('SUBMISSION.csv', 'a') as f:
        f.write('img,rle_mask\n')
        for index, i in enumerate(os.listdir(dir)):
            print('{}/{}'.format(index, N))

            img = Image.open(dir + i)

            mask = predict_img(net, img, gpu)
            enc = rle_encode(mask)
            f.write('{},{}\n'.format(i, ' '.join(map(str, enc))))


if __name__ == '__main__':
    net = UNet(3, 1).cuda()
    net.load_state_dict(torch.load('MODEL.pth'))
    submit(net, True)
