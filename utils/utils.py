import random
import numpy as np
import cv2


def get_square(img, pos):
    """Extract a left or a right square from ndarray shape : (H, W, C))"""
    h = img.shape[0]
    if pos == 0:
        return img[:, :h]
    else:
        return img[:, -h:]

def split_img_into_squares(img):
    return get_square(img, 0), get_square(img, 1)

def hwc_to_chw(img):
    trans_image = np.transpose(img, axes=[2, 0, 1])
    return trans_image

def resize_and_crop(pilimg, scale=0.5, final_height=None):
    w = pilimg.size[0]
    h = pilimg.size[1]
    newW = int(w * scale)
    newH = int(h * scale)

    if not final_height:
        diff = 0
    else:
        diff = newH - final_height

    img = pilimg.resize((newW, newH))
    img = img.crop((0, diff // 2, newW, newH - diff // 2))
    return np.array(img, dtype=np.float32)

def batch(iterable, batch_size):
    """Yields lists by batch"""
    b = []
    for i, t in enumerate(iterable):
        b.append(t)
        if (i + 1) % batch_size == 0:
            yield b
            b = []

    if len(b) > 0:
        yield b

def split_train_val(dataset, val_percent=0.05):
    dataset = list(dataset)
    length = len(dataset)
    n = int(length * val_percent)
    random.shuffle(dataset)
    trainset = dataset[:-n]
    valset = dataset[-n:]
    return {'train': trainset, 'val': valset}


def normalize(x):
    return x / 255

def merge_masks(img1, img2, full_w):
    
    h = img1.shape[1]
    new = np.zeros((h, full_w), np.float32)
    new[:, :full_w // 2 + 1] = img1[:, :full_w // 2 + 1]
    new[:, full_w // 2 + 1:] = img2[:, -(full_w // 2 - 1):]
    #return as np.float32
    return new


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


def split_mask_image(mask_image,class_mark):
    batch_size = mask_image.shape[0]
    n_classes = len(class_mark)
    split_mask = np.zeros((batch_size,n_classes - 1,mask_image.shape[1],mask_image.shape[2]),np.float32)
    for bs in range (batch_size):
        #except 0
        for i in range(1,n_classes):
            temp_mask = mask_image[bs].copy()
            temp_mask[temp_mask!=class_mark[i]]=0
            temp_mask = temp_mask/class_mark[i]
            split_mask[bs,i-1,:,:] = temp_mask
    return split_mask


