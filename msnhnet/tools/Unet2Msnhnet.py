
import torch
import torch.nn as nn
from PytorchToMsnhnet import *

import sys 
sys.path.append("../..") 

from unet.unet_model import UNet

net = UNet(3,2)
net.load_state_dict(torch.load("weights/MODEL.pth"))
net.to("cpu")
net.eval()

input=torch.ones([1,3,512,512])

trans(net, input,"unet.msnhnet","unet.msnhbin",False)