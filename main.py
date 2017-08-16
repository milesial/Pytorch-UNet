#models
from unet_model import UNet
from myloss import *
import torch
from torch.autograd import Variable
from torch import optim

#data manipulation
import numpy as np
import pandas as pd
import cv2
import PIL

#load files
import os

#data vis
from data_vis import plot_img_mask
from utils import *
import matplotlib.pyplot as plt


dir = 'data'
ids = []

for f in os.listdir(dir + '/train'):
    id = f[:-4]
    ids.append([id, 0])
    ids.append([id, 1])

np.random.shuffle(ids)
#%%


net = UNet(3, 1)

optimizer = optim.Adam(net.parameters(), lr=0.001)
criterion = DiceLoss()

dataset = []
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
        X = torch.FloatTensor(X / 255).unsqueeze(0)
        y = Variable(torch.ByteTensor(y))

        X = Variable(X, requires_grad=False)

        optimizer.zero_grad()

        y_pred = net(X).squeeze(1)


        loss = criterion(y_pred, y.unsqueeze(0).float())

        l += loss.data[0]
        loss.backward()
        optimizer.step()

        print('{0:.4f}%.'.format(i/len(ids)*100, end=''))

    print('Loss : {}'.format(l))


#%%




#net = UNet(3, 2)

#x = Variable(torch.FloatTensor(np.random.randn(1, 3, 640, 640)))

#y = net(x)


#plt.imshow(y[0])
#plt.show()
