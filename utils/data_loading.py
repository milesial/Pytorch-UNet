import logging
from os import listdir
from os.path import splitext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import copy


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img_ndarray = np.asarray(pil_img)

        if not is_mask:
            if img_ndarray.ndim == 2:
                img_ndarray = img_ndarray[np.newaxis, ...]
            else:
                img_ndarray = img_ndarray.transpose((2, 0, 1))

            img_ndarray = img_ndarray / 255

        return img_ndarray

    @staticmethod
    def load(filename):
        ext = splitext(filename)[1]
        if ext in ['.npz', '.npy']:
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            return Image.open(filename)

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.masks_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = self.load(mask_file[0])
        img = self.load(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale, is_mask=False)
        mask = self.preprocess(mask, self.scale, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, masks_dir, scale=1):
        super().__init__(images_dir, masks_dir, scale, mask_suffix='_mask')


class OCTADataset(Dataset):
    def __init__(self, data, dir_path, transform=None):
        self.data = data
        self.transform = transform
        self.dir_path = dir_path

        # load the whole dataset in memory
        self.samples = []

        for idx in range(len(data)):
            img_path = self.data[self.dir_path][idx]
            # img_path = os.path.join('data', img_path)

            md_10 = np.array([self.data["md_10"][idx]])
            psd_10 = np.array([self.data["psd_10"][idx]])

            td = self.data.iloc[idx, 78:146].to_numpy(dtype=np.float32)
            pd = self.data.iloc[idx, 146:214].to_numpy(dtype=np.float32)

            img = Image.open(img_path).convert("RGB")

            # sample = {'img': img, 'labels': np.concatenate((md_10, td, psd_10, pd))}
            sample = (img, np.concatenate((md_10, psd_10, td, pd)))

            self.samples.append(sample)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # img_path = self.data[self.dir_path][idx]
        # img_path = os.path.join('data', img_path)

        # md_10 = np.array([self.data["md_10"][idx]])
        # psd_10 = np.array([self.data["psd_10"][idx]])

        # td = self.data.iloc[idx, 78:146].to_numpy(dtype=np.float32)
        # pd = self.data.iloc[idx, 146:214].to_numpy(dtype=np.float32)

        # img = Image.open(img_path).convert("RGB")
        # sample = {'img': img, 'labels': np.concatenate((md_10, td, psd_10, pd))}

        sample = copy.deepcopy(self.samples[idx])

        if self.transform:
            # img = self.transform(img)
            # sample['img'] = self.transform(sample['img'])

            # sample[0] = self.transform(sample[0])

            # return (self.transform(sample[0]), sample[1])
            return self.transform(sample[0]), sample[1]
