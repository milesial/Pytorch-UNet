import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


def LoG_np(k, sigma):
    ax = np.round(np.linspace(-np.floor(k/2), np.floor(k/2), k))
    x, y = np.meshgrid(ax, ax)
    x2 = np.power(x, 2)
    y2 = np.power(y, 2)
    s2 = np.power(sigma, 2)
    s4 = np.power(sigma, 4)
    hg = np.exp(-(x2 + y2)/(2.*s2))
    kernel_t = hg*(x2 + y2-2*s2)/(s4*np.sum(hg))
    kernel = kernel_t - np.sum(kernel_t)/np.power(k, 2)
    return kernel


def LoG_2d(k, sigma, cuda=True):
    if cuda:
        ax = torch.round(torch.linspace(-math.floor(k/2),
                                        math.floor(k/2), k), out=torch.FloatTensor())
        ax = ax.cuda()
    else:
        ax = torch.round(torch.linspace(-math.floor(k/2),
                                        math.floor(k/2), k), out=torch.FloatTensor())
    y = ax.view(-1, 1).repeat(1, ax.size(0))
    x = ax.view(1, -1).repeat(ax.size(0), 1)
    x2 = torch.pow(x, 2)
    y2 = torch.pow(y, 2)
    s2 = torch.pow(sigma, 2)
    s4 = torch.pow(sigma, 4)
    #kernel = -(1./(math.pi*s4)) * (1.-(x2 + y2 / 2.*s2)) * (-(x2+y2)/(2.*s2)).exp()
    # return kernel / torch.sum(kernel)
    hg = (-(x2 + y2)/(2.*s2)).exp()
    kernel_t = hg*(x2 + y2-2*s2)/(s4*hg.sum())
    if cuda:
        kernel = kernel_t - kernel_t.sum() / \
            torch.pow(torch.FloatTensor([k]).cuda(), 2)
    else:
        kernel = kernel_t - kernel_t.sum() / \
            torch.pow(torch.FloatTensor([k]), 2)
    return kernel


class LoG2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, fixed_coeff=False, sigma=-1, stride=1, padding=0, dilation=1, cuda=True):
        super(LoG2d, self).__init__()
        self.fixed_coeff = fixed_coeff
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.cuda = cuda
        if not self.fixed_coeff:
            if self.cuda:
                self.sigma = nn.Parameter(
                    torch.cuda.FloatTensor(1), requires_grad=True)
            else:
                self.sigma = nn.Parameter(
                    torch.FloatTensor(1), requires_grad=True)
        else:
            if self.cuda:
                self.sigma = torch.cuda.FloatTensor([sigma])
            else:
                self.sigma = torch.FloatTensor([sigma])
            self.kernel = LoG_2d(self.kernel_size, self.sigma, self.cuda)
        self.init_weights()

    def init_weights(self):
        if not self.fixed_coeff:
            self.sigma.data.uniform_(0.0001, 0.9999)

    def forward(self, input):
        if not self.fixed_coeff:
            self.kernel = LoG_2d(self.kernel_size, self.sigma, self.cuda)
        kernel = self.kernel
        # kernel size is (out_channels, in_channels, h, w)
        kernel = kernel.repeat(self.out_channels, self.in_channels, 1).view(
            self.out_channels, self.in_channels, self.kernel_size, -1)
        # , stride=self.stride, padding=self.padding, dilation=self.dilation)
        res = F.conv2d(input, kernel, padding=self.padding)
        return res


class USMBase(LoG2d):

    def __init__(self, in_channels, kernel_size, fixed_coeff=False, sigma=-1, stride=1, dilation=1, cuda=True, batch_len=10):
        # Padding must be forced so output size is = to input size
        #Thus, in_channels = out_channels
        padding = int((stride*(in_channels-1)+((kernel_size-1)
                                               * (dilation-1))+kernel_size-in_channels)/2)
        super(USMBase, self).__init__(in_channels, in_channels,
                                      kernel_size, fixed_coeff, sigma, stride, padding, dilation, cuda)
        self.alpha = None
        self.batch_len = batch_len
        self.cont = 0

    def i_weights(self):
        super().init_weights()
        self.alpha.data.uniform_(0, 10)

    def forward(self, input):
        if self.cont == self.batch_len-1:
            print('Lambda del USM aprendido: ' + str(self.alpha.data))
            self.cont = 0
        else:
            self.cont += 1
        B = super().forward(input)

        maxB = torch.max(torch.abs(B))
        maxInput = torch.max(input)

        B = maxInput * (B/maxB)
        A = input + self.alpha * B

        A[A < 0.0] = 0.0
        A[A > 255.0] = 255.0

        return A


class USM(USMBase):
    def __init__(self, in_channels, kernel_size, fixed_coeff=False, sigma=-1, stride=1, dilation=1, cuda=True, batch_len=10):
        super(USM, self).__init__(in_channels, kernel_size,
                                  fixed_coeff, sigma, stride, dilation, cuda, batch_len)
        if self.cuda:
            self.alpha = nn.Parameter(
                torch.cuda.FloatTensor(1), requires_grad=True)
        else:
            self.alpha = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.i_weights()


class AdaptiveUSM(USMBase):
    def __init__(self, in_channels, in_side, kernel_size, fixed_coeff=False, sigma=-1, stride=1, dilation=1, cuda=True):
        super(AdaptiveUSM, self).__init__(in_channels, kernel_size,
                                          fixed_coeff, sigma, stride, dilation, cuda)
        if self.cuda:
            self.alpha = nn.Parameter(torch.cuda.FloatTensor(
                in_side, in_side), requires_grad=True)
        else:
            self.alpha = nn.Parameter(torch.FloatTensor(
                in_side, in_side), requires_grad=True)
        self.i_weights()
