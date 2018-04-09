#
# load.py : utils on generators / lists of ids to transform from strings to
#           cropped images and masks

import os
from functools import partial

import numpy as np
from PIL import Image

from .utils import resize_and_crop, get_square, normalize


def get_ids(dir):
    """Returns a list of the ids in the directory"""
    return (f[:-4] for f in os.listdir(dir))


def split_ids(ids, n=2):
    """Split each id in n, creating n tuples (id, k) for each id"""
    return ((id, i) for i in range(n) for id in ids)


def to_cropped_imgs(ids, dir, suffix):
    """From a list of tuples, returns the correct cropped img"""
    for id, pos in ids:
        im = resize_and_crop(Image.open(dir + id + suffix))
        yield get_square(im, pos)


def get_imgs_and_masks(ids, dir_img, dir_mask):
    """Return all the couples (img, mask)"""

    imgs = to_cropped_imgs(ids, dir_img, '.jpg')

    # need to transform from HWC to CHW
    imgs_switched = map(partial(np.transpose, axes=[2, 0, 1]), imgs)
    imgs_normalized = map(normalize, imgs_switched)

    masks = to_cropped_imgs(ids, dir_mask, '_mask.gif')

    return zip(imgs_normalized, masks)


def get_full_img_and_mask(id, dir_img, dir_mask):
    im = Image.open(dir_img + id + '.jpg')
    mask = Image.open(dir_mask + id + '_mask.gif')
    return np.array(im), np.array(mask)
