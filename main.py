import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import cv2

import torch
import torch.utils.data
from torch.utils.data.dataset import Dataset


import torch.nn as nn
import torchvision
from torchvision import transforms, datasets

from src.functions import *
import src.Dataset as ds

import src.network
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-e', type=int)
parser.add_argument('-b', action='store_true')
args = parser.parse_args()

# Set the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"{device}; device: {device}")


# Hyperparameters

# Load data
trainingTransforms = transforms.Compose([
    transforms.ToTensor()
])

dataset = ds.ImageDataset("dummyData/ColorfulOriginal", transform=trainingTransforms)

# Calculate 90%-10% for a split between train-test set
dsLen = len(dataset)
trainLen = int(dsLen * 0.9)
testLen = dsLen - trainLen

trainSet, testSet = torch.utils.data.random_split(dataset, [trainLen, testLen])

trainset = torch.utils.data.DataLoader(trainSet, batch_size=10, shuffle=True)
testingData = torch.utils.data.DataLoader(testSet, batch_size=10, shuffle=True)

model = src.network.eccv16()
if(device.type != 'cpu'):
  print(f"Pushing model to CUDA")
  model.cuda()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=0.0)


save_images = True
best_losses = 1e10
epochs = args.e
b = args.b
print(f"Epochs {epochs}")

# Train model
for epoch in range(epochs):
  # Train for one epoch, then validate
  train(trainset, model, criterion, optimizer, epoch, device, args.b)
  with torch.no_grad():
    losses = validate(testingData, model, criterion, save_images, epoch, device)
  # Save checkpoint and replace old best model if current model is better
  if losses < best_losses:
    best_losses = losses
    pathlib.Path('checkpoints').mkdir(parents=True, exist_ok=True) 
    torch.save(model.state_dict(), 'checkpoints/model-epoch-{}-losses-{:.3f}.pth'.format(epoch+1,losses))
