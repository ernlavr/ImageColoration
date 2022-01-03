import time
import numpy as np
import pathlib
import cv2
import torch

def to_rgb(grayscale_input, ab_input, save_path=None, save_name=None):
  '''Show/save rgb image from grayscale and ab channels
     Input save_path in the form {'grayscale': '/path/', 'colorized': '/path/'}'''
  # Check if grayscale/colorized dirs exist if not the save that shit duh ヽ(͡◕ ͜ʖ ͡◕)ﾉ 
  pathlib.Path(save_path['grayscale']).mkdir(parents=True, exist_ok=True) 
  pathlib.Path(save_path['colorized']).mkdir(parents=True, exist_ok=True) 

  # Converts tensors to numpy arrays
  npGray = grayscale_input.numpy()
  npAB = ab_input.numpy()

  npAB *= 255.0/npAB.max()
  npAB_scaled = npAB - 110

  # Merge luminance with AB
  LAB = np.concatenate((npGray, npAB), axis=0) 
  LAB = LAB.transpose((1, 2, 0)) # Shifts it from (3, 256, 256) to (256, 256, 3)

  # Convert to RGB colorspace and normalize (cv2.imwrite requires integers 0-255)
  RGB = cv2.cvtColor(LAB, cv2.COLOR_LAB2BGR)
  RGB = RGB/(RGB.max()/255.0)

  if save_path is not None and save_name is not None: 
    # Export image
    cv2.imwrite(f"{save_path['colorized']}RGBd{save_name}", RGB) 

class AverageMeter(object):
  """ Computes and stores the average and current value
      A handy class from the PyTorch ImageNet tutorial
      https://github.com/pytorch/examples/blob/master/imagenet/main.py#L199
  """
  
  def __init__(self):
    self.reset()
  def reset(self):
    self.val, self.avg, self.sum, self.count = 0, 0, 0, 0
  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count

def scaleTensor(tensor, float):
  """Scales a tensor's last two dimensions (colors) by the scalaer

  Args:
      tensor (torch.tensor): Tensor to scale
      scalar (float): scalar
  """
  npScalar = np.full(tensor.shape, float, dtype=np.float32)
  newTarget = tensor.detach().cpu() * npScalar[None, None, :, :]
  return newTarget[0, 0, :, :, :, :]


def train(train_loader, model, criterion, optimizer, epoch, device, brk):
  """Function for training the model

  Args:
      train_loader (Dataset): training dataset
      model (nn.module): Neural network
      criterion (torch.nn._Loss): Loss function
      optimizer (torch.nn.Optim): Optimizer
      epoch (int): Current epoch
      device (str): Either 'cpu' or 'gpu'
      brk (boolean): Break after first datapoint is processed. Used for debuggin on CPU
  """

  # Put the model in training mode
  model.train()
  print(f'Starting training epoch {epoch} and using device: {device}')
  
  # Prepare value counters and timers
  batch_time, data_time, losses = AverageMeter(), AverageMeter(), AverageMeter()

  end = time.time()
  for i, (target, input_gray, input_ab) in enumerate(train_loader):
    
    # Push the target, grayscale and AB tensors to CPU/GPU
    target      : torch.Tensor = target.to(device)
    input_gray  : torch.Tensor = input_gray.to(device)
    input_ab    : torch.Tensor = input_ab.to(device)   

    print(f"{torch.max(target)} {torch.min(target)}")
    print(f"{torch.max(input_gray)} {torch.min(input_gray)}")
    print(f"{torch.max(input_ab)} {torch.min(input_ab)}")

    target = scaleTensor(target, 255)
    input_gray = scaleTensor(input_gray, 255)
    input_ab = scaleTensor(input_ab, 255)

    print(f"{torch.max(target)} {torch.min(target)}")
    print(f"{torch.max(input_gray)} {torch.min(input_gray)}")
    print(f"{torch.max(input_ab)} {torch.min(input_ab)}")

    # Record time to load data (above)
    data_time.update(time.time() - end)

    # Run forward pass
    output_ab = model(input_gray) 
    loss = criterion(output_ab, input_ab) 
    losses.update(loss.item(), input_gray.size(0))

    # Compute gradient and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Record time to do forward and backward passes
    batch_time.update(time.time() - end)
    end = time.time()

    # Print model accuracy -- in the code below, val refers to value, not validation
    print('Epoch: [{0}][{1}/{2}]\t'
        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
            epoch, i, len(train_loader), batch_time=batch_time,
            data_time=data_time, loss=losses)) 
    
    # For quick debugging on CPU..
    if(brk and i == 1):
      print("Breaking after first iteration")
      break

  print('Finished training epoch {}'.format(epoch))

def validate(val_loader, model, criterion, save_images, epoch, device):
  model.eval()

  # Prepare value counters and timers
  batch_time, data_time, losses = AverageMeter(), AverageMeter(), AverageMeter()

  end = time.time()
  already_saved_images = False
  for i, (target, input_gray, input_ab) in enumerate(val_loader):
    data_time.update(time.time() - end)

    # Use GPU if available
    target = target.to(device)
    input_gray = input_gray.to(device)
    input_ab = input_ab.to(device)   


    # Run model and record loss
    output_ab = model(input_gray) # throw away class predictions
    loss = criterion(output_ab, input_ab)
    losses.update(loss.item(), input_gray.size(0))

    # Save images to file
    if save_images and not already_saved_images:
      already_saved_images = True
      for j in range(min(len(output_ab), 10)): # save at most 5 images
        save_path = {'grayscale': 'outputs/gray/', 'colorized': 'outputs/color/'}
        save_name = 'img-{}-epoch-{}.jpg'.format(i * val_loader.batch_size + j, epoch)
        to_rgb(input_gray[j].cpu(), ab_input=output_ab[j].detach().cpu(), save_path=save_path, save_name=save_name)

    # Record time to do forward passes and save images
    batch_time.update(time.time() - end)
    end = time.time()

    # Print model accuracy -- in the code below, val refers to both value and validation
    if i % 25 == 0:
      print('Validate: [{0}/{1}]\t'
            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
             i, len(val_loader), batch_time=batch_time, loss=losses))

  print('Finished validation.')
  return losses.avg
