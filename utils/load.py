#
# load.py : utils on generators / lists of ids to transform from strings to
#           cropped images and masks

import os

import numpy as np
from PIL import Image
import cv2

from .utils import resize_and_crop, get_square, normalize, hwc_to_chw


def get_ids(dir):
    """Returns a list of the ids in the directory"""
    return (f[:-4] for f in os.listdir(dir))


def split_ids(ids, n=2):
    """Split each id in n, creating n tuples (id, k) for each id"""
    return ((id, i) for i in range(n) for id in ids)


def to_cropped_imgs(ids, dir, suffix, scale):
    """From a list of tuples, returns the correct cropped img"""
    for id, pos in ids:
        im = resize_and_crop(Image.open(dir + id + suffix), scale=scale)
        yield get_square(im, pos)

def get_imgs_and_masks(ids, dir_img, dir_mask, scale):
    """Return all the couples (img, mask)"""
    #ids
    #tuple('image_name',0)
    imgs = to_cropped_imgs(ids, dir_img, '.png', scale)
    # need to transform from HWC to CHW
    imgs_switched = map(hwc_to_chw, imgs)
    imgs_normalized = map(normalize, imgs_switched)
    masks = to_cropped_imgs(ids, dir_mask, '_.png', scale)
    return zip(imgs_normalized, masks)


def get_full_img_and_mask(ids, dir_img, dir_mask,suffix_img, suffix_mask, target_size):
    #train as full size image
    list1 = []
    for id, pos in ids:
        im_p = Image.open(dir_img + id + suffix_img)
        mask_p = Image.open(dir_mask + id + suffix_mask)
        if(target_size[0]!=0):
            # PIL.resize(width,height)
            im_p = im_p.resize((target_size[1], target_size[0]))
            mask_p = mask_p.resize((target_size[1], target_size[0]))
        mask = np.array(mask_p, dtype=np.float32)
        im = np.array(im_p, dtype=np.float32)
        im = hwc_to_chw(im)
        list1.append((im, mask))
    return list1
