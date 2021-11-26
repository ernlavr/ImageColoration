import torch
import torch.utils.data
import torch.nn as nn
from torchvision import transforms
import argparse

import src.Dataset as ds
import src.network
from src.networkUtils import *


def parseArgs():
  # Custom args for debugging..
  parser = argparse.ArgumentParser()

  parser.add_argument('-e', type=int, help="Number of epochs")
  parser.add_argument('-b', action='store_true', help="Automatically break after first batch.. Used for debugging CPU output")
  return parser.parse_args()

def main():
  # Get args
  args = parseArgs()

  # Set either CPU or GPU as the computing engine
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print(f"Torch set to compute on device: {device}")


  # Specify transforms and initialize the dataset
  trainingTransforms = transforms.Compose([transforms.ToTensor()])
  dataset = ds.ImageDataset("dummyData/ColorfulOriginal", transform=trainingTransforms)

  # Calculate 90%-10% for a split between train-test set
  dsLen = len(dataset)
  trainLen = int(dsLen * 0.9)
  testLen = dsLen - trainLen

  # Split training and testing sets and create DataLoader object
  trainSet, testSet = torch.utils.data.random_split(dataset, [trainLen, testLen])
  trainset = torch.utils.data.DataLoader(trainSet, batch_size=10, shuffle=True)
  testingData = torch.utils.data.DataLoader(testSet, batch_size=10, shuffle=True)

  # Initialize Neural Net object and push it to CUDA if available
  model = src.network.eccv16()
  if(device.type != 'cpu'):
    print(f"Pushing model to CUDA")
    model.cuda()

  # Define loss function and optimizer
  criterion = nn.MSELoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=0.0)

  # Epochs, checkpoint condition, debugging..
  save_images = True
  best_losses = 1e10
  epochs = args.e
  b = args.b

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

if __name__ == '__main__':
  main()