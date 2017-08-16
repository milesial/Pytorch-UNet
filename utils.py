import PIL

def split_into_squares(img):
    """Extract a left and a right square from ndarray"""
    """shape : (H, W, C))"""
    h = img.shape[0]
    w = img.shape[1]


    left = img[:, :h]
    right = img[:, -h:]

    return left, right

def resize_and_crop(pilimg, scale=0.5, final_height=640):
    w = pilimg.size[0]
    h = pilimg.size[1]
    newW = int(w * scale)
    newH = int(h * scale)
    diff = newH - final_height

    img = pilimg.resize((newW, newH))
    img = img.crop((0, diff // 2, newW, newH - diff // 2))
    return img
