import torch
from utils import *
import torch.nn.functional as F
from PIL import Image
from unet_model import UNet
from torch.autograd import Variable
import matplotlib.pyplot as plt
from crf import dense_crf


def predict_img(net, full_img, gpu=False):
    img = resize_and_crop(full_img)

    left = get_square(img, 0)
    right = get_square(img, 1)

    right = normalize(right)
    left = normalize(left)

    right = np.transpose(right, axes=[2, 0, 1])
    left = np.transpose(left, axes=[2, 0, 1])

    X_l = torch.FloatTensor(left).unsqueeze(0)
    X_r = torch.FloatTensor(right).unsqueeze(0)

    if gpu:
        X_l = Variable(X_l, volatile=True).cuda()
        X_r = Variable(X_r, volatile=True).cuda()
    else:
        X_l = Variable(X_l, volatile=True)
        X_r = Variable(X_r, volatile=True)

    y_l = F.sigmoid(net(X_l))
    y_r = F.sigmoid(net(X_r))
    y_l = F.upsample_bilinear(y_l, scale_factor=2).data[0][0].cpu().numpy()
    y_r = F.upsample_bilinear(y_r, scale_factor=2).data[0][0].cpu().numpy()

    y = merge_masks(y_l, y_r, 1918)
    yy = dense_crf(np.array(full_img).astype(np.uint8), y)

    return yy > 0.5
