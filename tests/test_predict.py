import unittest

import numpy as np
import torch
from PIL import Image

from predict import predict_img
from unet import UNet


class PredictImageModeTests(unittest.TestCase):
    def test_three_channel_models_accept_non_rgb_images(self):
        pixels = np.random.default_rng(0).integers(0, 256, (32, 48, 3), dtype=np.uint8)
        rgb_image = Image.fromarray(pixels)
        palette_image = Image.fromarray(pixels[:, :, 0]).convert('P')
        palette_image.putpalette([value for i in range(256) for value in (i, 255 - i, i // 2)])
        rgba_image = rgb_image.convert('RGBA')
        rgba_image.putalpha(0)
        images = [rgb_image, rgba_image, rgb_image.convert('L'), palette_image]

        for n_classes, bilinear in ((2, False), (1, True)):
            torch.manual_seed(0)
            net = UNet(n_channels=3, n_classes=n_classes, bilinear=bilinear)
            for image in images:
                original_mode = image.mode
                for scale in (0.5, 1.0):
                    with self.subTest(classes=n_classes, mode=image.mode, scale=scale):
                        expected = predict_img(net, image.convert('RGB'), torch.device('cpu'), scale)
                        actual = predict_img(net, image, torch.device('cpu'), scale)
                        self.assertEqual(actual.shape, (32, 48))
                        np.testing.assert_array_equal(actual, expected)
                        self.assertEqual(image.mode, original_mode)

    def test_one_channel_model_keeps_grayscale_input(self):
        image = Image.fromarray(np.full((32, 48), 128, dtype=np.uint8))
        net = UNet(n_channels=1, n_classes=2)
        mask = predict_img(net, image, torch.device('cpu'))
        self.assertEqual(mask.shape, (32, 48))

    def test_four_channel_model_keeps_rgba_input(self):
        image = Image.new('RGBA', (48, 32), (64, 128, 192, 255))
        net = UNet(n_channels=4, n_classes=2)
        mask = predict_img(net, image, torch.device('cpu'))
        self.assertEqual(mask.shape, (32, 48))


if __name__ == '__main__':
    unittest.main()
