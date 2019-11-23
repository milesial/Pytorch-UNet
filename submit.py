""" Submit code specific to the kaggle challenge"""

import os

import torch
from PIL import Image
import numpy as np

from predict import predict_img
from unet import UNet

# credits to https://stackoverflow.com/users/6076729/manuel-lagunas
def rle_encode(mask_image):
    pixels = mask_image.flatten()
    # We avoid issues with '1' at the start or end (at the corners of
    # the original image) by setting those pixels to '0' explicitly.
    # We do not expect these to be non-zero for an accurate mask,
    # so this should not harm the score.
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    return runs


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
