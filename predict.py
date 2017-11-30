import torch
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy
from PIL import Image
import argparse
import os

from utils import *

from unet import UNet


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

    y = merge_masks(y_l, y_r, full_img.size[0])
    yy = dense_crf(np.array(full_img).astype(np.uint8), y)

    return yy > 0.5


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='MODEL.pth',
                        metavar='FILE',
                        help="Specify the file in which is stored the model"
                        " (default : 'MODEL.pth')")
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+',
                        help='filenames of input images', required=True)
    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+',
                        help='filenames of ouput images')
    parser.add_argument('--cpu', '-c', action='store_true',
                        help="Do not use the cuda version of the net",
                        default=False)
    parser.add_argument('--viz', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=False)
    parser.add_argument('--no-save', '-n', action='store_false',
                        help="Do not save the output masks",
                        default=False)

    args = parser.parse_args()
    print("Using model file : {}".format(args.model))
    net = UNet(3, 1)
    if not args.cpu:
        print("Using CUDA version of the net, prepare your GPU !")
        net.cuda()
    else:
        net.cpu()
        print("Using CPU version of the net, this may be very slow")

    in_files = args.input
    out_files = []
    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
    elif len(in_files) != len(args.output):
        print("Error : Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output

    print("Loading model ...")
    net.load_state_dict(torch.load(args.model))
    print("Model loaded !")

    for i, fn in enumerate(in_files):
        print("\nPredicting image {} ...".format(fn))
        img = Image.open(fn)
        out = predict_img(net, img, not args.cpu)

        if args.viz:
            print("Vizualising results for image {}, close to continue ..."
                  .format(fn))

            fig = plt.figure()
            a = fig.add_subplot(1, 2, 1)
            a.set_title('Input image')
            plt.imshow(img)

            b = fig.add_subplot(1, 2, 2)
            b.set_title('Output mask')
            plt.imshow(out)

            plt.show()

        if not args.no_save:
            out_fn = out_files[i]
            result = Image.fromarray((out * 255).astype(numpy.uint8))
            result.save(out_files[i])
            print("Mask saved to {}".format(out_files[i]))
