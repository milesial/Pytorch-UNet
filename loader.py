from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import cv2
import numpy as np
import os
import random


class BBBCDataset(Dataset):
    def __init__(self, ids, dir_data, dir_gt, extension='.png', gt_label='_mask'):

        self.dir_data = dir_data
        self.dir_gt = dir_gt
        self.extension = extension
        self.gt_label = gt_label

        # Transforms
        self.transformations = transforms.Compose([transforms.ToTensor()])

        # Images IDS
        self.ids = ids

        # Calculate len of data
        self.data_len = len(self.ids)

    def __getitem__(self, index):
        # Get an ID of a specific image
        id_img = self.dir_data + self.ids[index] + self.extension
        id_gt = self.dir_gt + self.ids[index] + self.gt_label + self.extension
        # Open Image and GroundTruth
        img = np.array(cv2.imread(id_img)).astype(np.float32)
        gt = np.array(cv2.imread(id_gt,0)).astype(np.float32)

        # Applies transformations
        img = self.transformations(img)

        return (img, gt)

    def __len__(self):
        return self.data_len


def get_dataloaders(dir_img, dir_gt, test_percent=0.2, batch_size=10):
    # Validate a correct percentage
    test_percent = test_percent/100 if test_percent > 1 else test_percent
    # Read the names of the images
    ids = [f[:-4] for f in os.listdir(dir_img)]
    # Rearrange the images
    random.shuffle(ids)
    # Calculate index of partition
    part = int(len(ids) * test_percent)

    # Split dataset between train and test
    train_ids = ids[part:]
    test_ids = ids[:part]

    # Create the datasets
    train_dataset = BBBCDataset(ids=train_ids, dir_data=dir_img, dir_gt=dir_gt)
    test_dataset = BBBCDataset(ids=test_ids, dir_data=dir_img, dir_gt=dir_gt)

    # Create the loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader
