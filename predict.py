import torch
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy
from PIL import Image
import argparse
import os
import torchvision
import numpy as np
from utils import *
import scipy.io

from unet import UNet


def predict_img(net, full_img, gpu=False):
    img = numpy.array(resize(full_img))
    img = normalize(img)
    img = np.transpose(img, axes=[2,0,1])
    X = torch.FloatTensor(img).unsqueeze(0)

    if gpu:
        X = Variable(X, volatile=True).cuda()
    else:
        X = Variable(X, volatile=True)

    y = net(X)
    return y

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
    parser.add_argument('--depth-map', '-d', action='store_true',
                        help="Model based on depth-map instead of RGB",
                        default=True)

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

        out_fn = out_files[i]
        if args.depth_map:
            test = out.data.squeeze().cpu().numpy()
            scipy.io.savemat(out_fn, {"data":test})
        else:
            torchvision.utils.save_image(out.data, out_fn)
            # result = Image.fromarray((out * 255).astype(numpy.uint8))
            # result.save(out_files[i])

        print("Mask saved to {}".format(out_files[i]))
