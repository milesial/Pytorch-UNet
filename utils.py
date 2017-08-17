import PIL
import numpy as np

def get_square(img, pos):
    """Extract a left or a right square from PILimg"""
    """shape : (H, W, C))"""
    img = np.array(img)

    h = img.shape[0]
    w = img.shape[1]

    if pos == 0:
        return img[:, :h]
    else:
        return img[:, -h:]

def resize_and_crop(pilimg, scale=0.5, final_height=640):
    w = pilimg.size[0]
    h = pilimg.size[1]
    newW = int(w * scale)
    newH = int(h * scale)
    diff = newH - final_height

    img = pilimg.resize((newW, newH))
    img = img.crop((0, diff // 2, newW, newH - diff // 2))
    return img
