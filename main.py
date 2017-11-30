#models
from unet import UNet
from myloss import *
import torch
from torch.autograd import Variable
from torch import optim

#data manipulation
import numpy as np
import pandas as pd
import PIL

#load files
import os

#data visualization
from data_vis import plot_img_mask
from utils import *
import matplotlib.pyplot as plt

#quit after interrupt
import sys



dir = 'data'
ids = []

for f in os.listdir(dir + '/train'):
    id = f[:-4]
    ids.append([id, 0])
    ids.append([id, 1])

np.random.shuffle(ids)
#%%


net = UNet(3, 1)
net.cuda()

def train(net):
    optimizer = optim.Adam(net.parameters(), lr=1)
    criterion = DiceLoss()

    epochs = 5
    for epoch in range(epochs):
        print('epoch {}/{}...'.format(epoch+1, epochs))
        l = 0

        for i, c in enumerate(ids):
            id = c[0]
            pos = c[1]
            im = PIL.Image.open(dir + '/train/' + id + '.jpg')
            im = resize_and_crop(im)

            ma = PIL.Image.open(dir + '/train_masks/' + id + '_mask.gif')
            ma = resize_and_crop(ma)

            left, right = split_into_squares(np.array(im))
            left_m, right_m = split_into_squares(np.array(ma))

            if pos == 0:
                X = left
                y = left_m
            else:
                X = right
                y = right_m


            X = np.transpose(X, axes=[2, 0, 1])
            X = torch.FloatTensor(X / 255).unsqueeze(0).cuda()
            y = Variable(torch.ByteTensor(y)).cuda()

            X = Variable(X).cuda()

            optimizer.zero_grad()

            y_pred = net(X).squeeze(1)


            loss = criterion(y_pred, y.unsqueeze(0).float())

            l += loss.data[0]
            loss.backward()
            if i%10 == 0:
                optimizer.step()
                print('Stepped')

            print('{0:.4f}%\t\t{1:.6f}'.format(i/len(ids)*100, loss.data[0]))

        l = l / len(ids)
        print('Loss : {}'.format(l))
        torch.save(net.state_dict(), 'MODEL_EPOCH{}_LOSS{}.pth'.format(epoch+1, l))
        print('Saved')

try:
    net.load_state_dict(torch.load('MODEL_INTERRUPTED.pth'))
    train(net)

except KeyboardInterrupt:
    print('Interrupted')
    torch.save(net.state_dict(), 'MODEL_INTERRUPTED.pth')
    try:
        sys.exit(0)
    except SystemExit:
        os._exit(0)
