import os
import cv2
import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
from src.Utilities import *

import torch
from torch.utils.data import Dataset
from torchvision import datasets

from src.image import Image

import skimage
import skimage.transform
from skimage import io
from skimage.color import rgb2lab, rgb2gray

class ImageDataset(Dataset):
    """
    Wrapper class for importing the custom data
    """

    def __init__(self,
                 imgDir,
                 return_labels=True,
                 return_filenames=False,
                 resize=256,
                 transform=None):

        self.imgDir : str   = os.path.join(os.getcwd(), imgDir)
        self.images : Image = []
        self.transform = transform
        self.dim = resize
        self.populateImageList()

    def populateImageList(self):
        """Recursively populates images collection from the given image directory
        """
        
        for root, directories, files in os.walk(self.imgDir):
            for file in files:
                filePath = os.path.join(root, file)

                # Read and resize original RGB pic
                img = cv2.imread(filePath)
                img = cv2.resize(img, (self.dim, self.dim))

                # Get Gray and LAB versions
                imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                imgLAB = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                L, A, B = cv2.split(imgLAB)
                AB = cv2.merge((A, B))

                # Fill an entry with RGB, imgGray and AB target
                # To get original image, merge imgGray + AB -> LAB2BGR
                entry = Image(file, img, imgGray, AB)
                self.images.append(entry)
        self.images = np.asarray(self.images)


    def __len__(self):
        return len(self.images)


    def __getitem__(self, index):
        if self.transform is not None:
            name = self.images[index].Name
            rgb = self.images[index].RGB
            gray = self.images[index].Gray
            ab = self.images[index].AB

            rgb = self.transform(rgb)
            gray = self.transform(gray)
            ab = self.transform(ab)

            return name, rgb, gray, ab