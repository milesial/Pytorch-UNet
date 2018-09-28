import argparse

import torch
from torchvision import transforms
import numpy as np
from matplotlib import pyplot as plt
import cv2

from unet import UNet
from loader import get_predloader


def predict_imgs(net, device, loader, dir_save, show=False):
    with torch.no_grad():
        for batch_idx, (data, gt) in enumerate(loader):

            # Use GPU or not
            data = data.to(device)

            if show:
                # Shows original image
                data_img = transforms.ToPILImage()(data.squeeze(0).cpu()).convert('RGBA')
                fig = plt.figure(figsize=(20, 20))
                fig.add_subplot(1, 4, 1)
                plt.imshow(data_img)

            # Forward
            predictions = net(data)

            # Apply sigmoid
            pred_probs = torch.sigmoid(predictions).squeeze(0)

            # Shows prediction
            if show:
                # Shows prediction
                pred = transforms.ToPILImage()(predictions.squeeze(0).cpu()).convert('RGBA')
                fig.add_subplot(1, 4, 2)
                plt.imshow(pred)
                # Shows prediction probability
                pred_p = transforms.ToPILImage()(pred_probs.cpu()).convert('RGBA')
                fig.add_subplot(1, 4, 3)
                plt.imshow(pred_p)
                # Shows gt
                gt_img = transforms.ToPILImage()(gt.squeeze(0).cpu()).convert('RGBA')
                fig.add_subplot(1, 4, 4)
                plt.imshow(gt_img)
                plt.show()


def predict(load='checkpoints/CP1.pth'):

    # Use GPU or not
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Create the model
    net = UNet(n_channels=1, n_classes=1).to(device)

    # Load trained weights
    net.load_state_dict(torch.load(load))
    print('Model loaded from {}'.format(load))

    # Location of the images to use
    dir_pred = 'data/to_predict/'
    dir_save = 'data/predicted/'

    # Load the dataset
    pred_loader = get_predloader(dir_pred)

    # Run the prediction
    predict_imgs(net=net,
                 device=device,
                 loader=pred_loader,
                 dir_save=dir_save,
                 show=True)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='checkpoints/CP1.pth',
                        metavar='FILE',
                        help="Specify the file in which is stored the model"
                             " (default : 'checkpoints/CP1.pth')")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    predict(load=args.model)
