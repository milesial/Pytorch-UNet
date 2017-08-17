import torch

from load import *
from data_vis import *
from utils import split_train_val, batch
from myloss import DiceLoss
from unet_model import UNet
from torch.autograd import Variable
from torch import optim
from optparse import OptionParser


def train_net(net, epochs=5, batch_size=2, lr=0.1, val_percent=0.05,
              cp=True, gpu=False):
    dir_img = 'data/train/'
    dir_mask = 'data/train_masks/'
    dir_checkpoint = 'checkpoints/'

    # get ids
    ids = get_ids(dir_img)
    ids = split_ids(ids)

    iddataset = split_train_val(ids, val_percent)

    print('''
    Starting training:
        Epochs: {}
        Batch size: {}
        Learning rate: {}
        Training size: {}
        Validation size: {}
        Checkpoints: {}
        CUDA: {}
    '''.format(epochs, batch_size, lr, len(iddataset['train']),
               len(iddataset['val']), str(cp), str(gpu)))

    N_train = len(iddataset['train'])

    train = get_imgs_and_masks(iddataset['train'], dir_img, dir_mask)
    val = get_imgs_and_masks(iddataset['val'], dir_img, dir_mask)

    optimizer = optim.Adam(net.parameters(), lr=lr)
    criterion = DiceLoss()

    for epoch in range(epochs):
        print('Starting epoch {}/{}.'.format(epoch+1, epochs))

        epoch_loss = 0

        for i, b in enumerate(batch(train, batch_size)):
            X = np.array([i[0] for i in b])
            y = np.array([i[1] for i in b])

            X = torch.FloatTensor(X)
            y = torch.ByteTensor(y)

            if gpu:
                X = Variable(X).cuda()
                y = Variable(y).cuda()
            else:
                X = Variable(X)
                y = Variable(y)

            optimizer.zero_grad()

            y_pred = net(X)

            loss = criterion(y_pred, y.float())
            epoch_loss += loss.data[0]

            print('{0:.4f} --- loss: {1:.6f}'.format(i*batch_size/N_train,
                                              loss.data[0]))

            loss.backward()
            optimizer.step()

        print('Epoch finished ! Loss: {}'.format(epoch_loss/i))

        if cp:
            torch.save(net.state_dict(),
                       dir_checkpoint + 'CP{}.pth'.format(epoch+1))

            print('Checkpoint {} saved !'.format(epoch+1))


parser = OptionParser()
parser.add_option("-e", "--epochs", dest="epochs", default=5, type="int",
                  help="number of epochs")
parser.add_option("-b", "--batch-size", dest="batchsize", default=10,
                  type="int", help="batch size")
parser.add_option("-l", "--learning-rate", dest="lr", default=0.1,
                  type="int", help="learning rate")
parser.add_option("-g", "--gpu", action="store_true", dest="gpu",
                  default=False, help="use cuda")
parser.add_option("-n", "--ngpu", action="store_false", dest="gpu",
                  default=False, help="use cuda")


(options, args) = parser.parse_args()

net = UNet(3, 1)
if options.gpu:
    net.cuda()

train_net(net, options.epochs, options.batchsize, options.lr, gpu=options.gpu)
