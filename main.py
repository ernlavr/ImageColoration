import torch
import torch.utils.data
import torch.nn as nn
from torchvision import transforms
import argparse
import os

import src.Dataset as ds
import src.network
from src.networkUtils import *


def parseArgs():
  # Custom args for debugging..
  parser = argparse.ArgumentParser()

  parser.add_argument('-e', type=int, help="Number of epochs")
  parser.add_argument('-b', action='store_true', help="Automatically break after first batch.. Used for debugging CPU output")
  parser.add_argument('--ts', type=str, help="Specify relative path to a training set folder")
  parser.add_argument('--vs', type=str, help="Specify relative path to a validation set folder")
  parser.add_argument('--pretrained', type=str, help="Indicates to use a pre-trained weight")
  parser.add_argument('--skip_training', action='store_true', help="Proceed straight to validation")
  return parser.parse_args()

def getDatasets(args):
  """ Returns training/testing dataset. Either it splits the data contained in dummyData/ColorfulOriginal
      or uses pre-specified folders depending
  """

  # Initialize the dataset
  trainingTransforms = transforms.Compose([transforms.ToTensor()])
  dataset = ds.ImageDataset("dummyData/NC_Dataset", transform=trainingTransforms)

  # Calculate 90%-10% for a split between train-test set
  dsLen = len(dataset)
  trainLen = int(dsLen * 0.9)
  valLen = dsLen - trainLen

  # Split training and testing sets and create DataLoader object
  trainSet, valSet = torch.utils.data.random_split(dataset, [trainLen, valLen])
  trainSet = torch.utils.data.DataLoader(trainSet, batch_size=10, shuffle=True)
  valSet = torch.utils.data.DataLoader(valSet, batch_size=10, shuffle=True)

  # Overwrite the dummy-data with pre-specified datasets if applicable. Inefficient but simple..
  if args.ts is not None:
    if(os.path.exists(args.ts)):
      set = ds.ImageDataset(args.ts, transform=trainingTransforms)
      trainSet = torch.utils.data.DataLoader(set, batch_size=10, shuffle=True)
      print(f"Loaded training set from {args.ts}")
  
  if args.vs is not None:
    if(os.path.exists(args.vs)):
      set = ds.ImageDataset(args.vs, transform=trainingTransforms)
      valSet = torch.utils.data.DataLoader(set, batch_size=10, shuffle=True)
      print(f"Loaded validation set from {args.vs}")

  return trainSet, valSet


def getModel(args):
  """Loads a model with either pretrained weights if framework is started with "--pretrained" flag;
     or blank model if without
  """
  model = src.network.eccv16()
  if args.pretrained is not None:
    if(os.path.exists(args.pretrained)):
      if(torch.cuda.is_available() == False):  # Load in CPU mode
        model.load_state_dict(torch.load(args.pretrained, map_location=torch.device('cpu')))  
      else:                                    # Load in normal mode
        model.load_state_dict(torch.load(args.pretrained))
      print(f"Loaded pretrained weights from path {args.pretrained}")
  
  return model
    

def main():
  # Get args
  args = parseArgs()

  # Set either CPU or GPU as the computing engine
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print(f"Torch set to compute on device: {device}")

  # Get datasets
  trainSet, validationSet = getDatasets(args)
  
  # Initialize Neural Net object and push it to CUDA if available
  model = getModel(args)
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

  # Just run the model and generate output..
  if(args.skip_training is True):
    with torch.no_grad():
      losses = validate(validationSet, model, criterion, save_images, 0, device)
  
  # Train-Validate-Save...
  else:
    for epoch in range(epochs):
      # Train for one epoch, then validate
      train(trainSet, model, criterion, optimizer, epoch, device, args.b)
      with torch.no_grad():
        losses = validate(validationSet, model, criterion, save_images, epoch, device)

      # Save checkpoint and replace old best model if current model is better
      if losses < best_losses:
        best_losses = losses
        pathlib.Path('checkpoints').mkdir(parents=True, exist_ok=True) 
        torch.save(model.state_dict(), 
                  'checkpoints/model-epoch-{}-losses-{:.3f}.pth'.format(epoch+1,losses))
    

if __name__ == '__main__':
  main()