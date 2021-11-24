import os
import cv2
import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset
from torchvision import datasets

from src.image import Image

import skimage
import skimage.transform
from skimage import io
from skimage.color import rgb2lab, rgb2gray

class ImageDataset(Dataset):
    """[summary] Wrapper class for importing the custom data
    """

    def __init__(self,
                 imgDir,
                 return_labels=True,
                 return_filenames=False,
                 resize=256,
                 transform=None):

        self.imgDirFullPath : str   = os.path.join(os.getcwd(), imgDir)
        self.images = []
        self.transform = transform
        self.dim = resize
        self.populateImageList()

    def populateImageList(self):
        for root, directories, files in os.walk(self.imgDirFullPath):
            for file in files:
                filePath = os.path.join(root, file)

                img = cv2.imread(filePath)
                img = cv2.resize(img, (self.dim, self.dim))

                imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                img1 = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
                AB = img[:, :, 1:3]

                entry = Image(img, imgGray, img[:, :, 1:3])
                self.images.append(entry)
        self.images = np.asarray(self.images)


    def __len__(self):
        return len(self.images)


    def __getitem__(self, index):
        if self.transform is not None:
            rgb = self.images[index].colorData
            gray = self.images[index].grayData
            ab = self.images[index].AB

            rgb = self.transform(rgb)
            gray = self.transform(gray)
            ab = self.transform(ab)

            return rgb, gray, ab