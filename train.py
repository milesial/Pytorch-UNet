import sys
import os
from optparse import OptionParser
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from unet import UNet
from loader import get_dataloaders


def train_net(net, device, loader, dir_checkpoint, epochs=5):
    ''' Train the CNN. '''
    for epoch in range(epochs):
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))

        net.train()
        train_loss = 0
        for batch_idx, (data, gt) in enumerate(loader):

            # Use GPU or not
            data, gt = data.to(device), gt.to(device)

            optimizer.zero_grad()

            # Forward
            predictions = net(data)

            # To calculate Loss
            pred_probs = F.sigmoid(predictions)
            pred_probs_flat = pred_probs.view(-1)
            gt_flat = gt.view(-1)

            # Loss Calculation
            loss = criterion(pred_probs_flat, gt_flat)
            train_loss += loss.item()

            # Backpropagation
            loss.backward()
            optimizer.step()

            print('{0:.4f} --- Training Loss: {1:.6f}'.format(100. *
                                                              batch_idx / len(train_loader), loss.item()))

        torch.save(net.state_dict(), dir_checkpoint +
                   'CP{}.pth'.format(epoch + 1))
        print('Checkpoint {} saved !'.format(epoch + 1))


def test_net(net, device, loader):
    ''' Test the CNN '''
    net.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, gt in loader:

            # Use GPU or not
            data, gt = data.to(device), gt.to(device)

            # Forward
            predictions = net(data)

            # To calculate Loss
            pred_probs = F.sigmoid(predictions)
            pred_probs_flat = pred_probs.view(-1)
            gt_flat = gt.view(-1)

            # Loss Calculation
            loss = criterion(pred_probs_flat, gt_flat)
            test_loss += loss.item()

            # get the index of the max log-probability
            pred = predictions.max(1, keepdim=True)[1]
            correct += pred.eq(gt.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=5, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batchsize', default=10,
                      type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.1,
                      type='float', help='learning rate')
    parser.add_option('-t', '--test-percentage', type='float', dest='testperc',
                      default=0.2, help='Test percentage')
    parser.add_option('-c', '--load', dest='load',
                      default=False, help='load file model')
    parser.add_option('-s', '--scale', dest='scale', type='float',
                      default=0.5, help='downscaling factor of the images')

    (options, args) = parser.parse_args()
    return options


if __name__ == '__main__':
    args = get_args()

    # Load old weights
    if args.load:
        net.load_state_dict(torch.load(args.load))
        print('Model loaded from {}'.format(args.load))

    # Use GPU or not
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # Create the model
    net = UNet(n_channels=3, n_classes=1).to(device)

    # Location of the images to use
    dir_img = 'data/train/'
    dir_gt = 'data/gt/'
    dir_checkpoint = 'checkpoints/'

    # Load the dataset
    train_loader, test_loader = get_dataloaders(
        dir_img, dir_gt, args.testperc, args.batchsize)

    # Pretty print of the run
    print('''
    Starting training:
        Epochs: {}
        Batch size: {}
        Learning rate: {}
        Training size: {}
        Testing size: {}
        CUDA: {}
    '''.format(args.epochs, args.batchsize, args.lr, len(train_loader.dataset),
               len(test_loader.dataset), str(use_cuda)))

    # Definition of the optimizer
    optimizer = optim.SGD(net.parameters(),
                          lr=args.lr,
                          momentum=0.9,
                          weight_decay=0.0005)

    # Definition of the loss function
    criterion = nn.BCELoss()

    # Run the training and testing
    try:
        train_net(net=net,
                  epochs=args.epochs,
                  device=device,
                  dir_checkpoint=dir_checkpoint,
                  loader=train_loader)
        test_net(net=net, device=device, loader=test_loader)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
