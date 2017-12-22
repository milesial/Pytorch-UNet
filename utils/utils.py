import PIL
import numpy as np
import random


def get_square(img, pos):
    """Extract a left or a right square from PILimg shape : (H, W, C))"""
    img = np.array(img)
    h = img.shape[0]
    if pos == 0:
        return img[:, :h]
    else:
        return img[:, -h:]


# def resize_and_crop(pilimg, scale=0.5, final_height=None):
def resize(pilimg, scale=0.5):
    w = pilimg.size[0]
    h = pilimg.size[1]
    newW = int(w * scale)
    newH = int(h * scale)

    # if not final_height:
    #     diff = 0
    # else:
    #     diff = newH - final_height

    img = pilimg.resize((newW, newH))
    # img = img.crop((0, diff // 2, newW, newH - diff // 2))
    return img


def batch(iterable, batch_size):
    """Yields lists by batch"""
    b = []
    for i, t in enumerate(iterable):
        b.append(t)
        if (i+1) % batch_size == 0:
            yield b
            b = []

    if len(b) > 0:
        yield b


def split_train_val(dataset, val_percent=0.05):
    dataset = list(dataset)
    length = len(dataset)
    n = int(length * val_percent)
    random.seed(42)
    random.shuffle(dataset)
    return {'train': dataset[:-n], 'val': dataset[-n:]}


def normalize(x):
    return x / 255


def merge_masks(img1, img2, full_w):
    w = img1.shape[1]
    overlap = int(2 * w - full_w)
    h = img1.shape[0]

    new = np.zeros((h, full_w), np.float32)

    margin = 0

    new[:, :full_w//2+1] = img1[:, :full_w//2+1]
    new[:, full_w//2+1:] = img2[:, -(full_w//2-1):]
    #new[:, w-overlap+1+margin//2:-(w-overlap+margin//2)] = (img1[:, -overlap+margin:] +
    #                                  img2[:, :overlap-margin])/2

    return new


import matplotlib.pyplot as plt

def encode(mask):
    """mask : HxW"""
    plt.imshow(mask.transpose())
    plt.show()
    flat = mask.transpose().reshape(-1)
    enc = []
    i = 1

    while i <= len(flat):
        if(flat[i-1]):
            s = i
            while(flat[i-1]):
                i += 1
            e = i-1
            enc.append(s)
            enc.append(e - s + 1)
        i += 1

    plt.imshow(decode(enc))
    plt.show()
    return enc

def decode(list):
    mask = np.zeros((1280*1920), np.bool)

    for i, e in enumerate(list):
        if(i%2 == 0):
            mask[e-1:e-2+list[i+1]] = True

    mask = mask.reshape(1920, 1280).transpose()

    return mask


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

def full_process(filename):
    im = PIL.Image.open(filename)
    im = resize_and_crop(im)
