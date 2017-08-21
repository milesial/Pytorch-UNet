import os
from PIL import Image
from predict import *
from utils import encode
from unet_model import UNet

def submit(net, gpu=False):
    dir = 'data/test/'

    N = len(list(os.listdir(dir)))
    with open('SUBMISSION.csv', 'w') as f:
        f.write('img,rle_mask\n')
        for index, i in enumerate(os.listdir(dir)):
            print('{}/{}'.format(index, N))
            img = Image.open(dir + i)

            mask = predict_img(net, img, gpu)
            enc = encode(mask)
            f.write('{},{}\n'.format(i, ' '.join(map(str, enc))))


if __name__ == '__main__':
    net = UNet(3, 1).cuda()
    net.load_state_dict(torch.load('INTERRUPTED.pth'))
    submit(net, True)
