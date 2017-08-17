import os
import random
import numpy as np
from PIL import Image
from functools import partial
from utils import resize_and_crop, get_square


def get_ids(dir):
    """Returns a list of the ids in the directory"""
    return (f[:-4] for f in os.listdir(dir))

def split_ids(ids, n=2):
    """Split each id in n, creating n tuples (id, k) for each id"""
    return ((id, i) for i in range(n) for id in ids)

def shuffle_ids(ids):
    """Returns a shuffle list od the ids"""
    lst = list(ids)
    random.shuffle(lst)
    return lst

def to_cropped_imgs(ids, dir, suffix):
    """From a list of tuples, returns the correct cropped img (left or right)"""
    for id, pos in ids:
        im = resize_and_crop(Image.open(dir + id + suffix))
        yield get_square(im, pos)



def get_imgs_and_masks():
    """From the list of ids, return the couples (img, mask)"""
    dir_img = 'data/train/'
    dir_mask = 'data/train_masks/'

    ids = get_ids(dir_img)
    ids = split_ids(ids)
    ids = shuffle_ids(ids)

    imgs = to_cropped_imgs(ids, dir_img, '.jpg')

    # need to transform from HWC to CHW
    imgs_switched = map(partial(np.transpose, axes=[2, 0, 1]), imgs)

    masks = to_cropped_imgs(ids, dir_mask, '_mask.gif')

    return zip(imgs_switched, masks)
