import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import pathlib
from skimage.color import lab2rgb, rgb2lab, rgb2gray
import cv2

def to_rgb(grayscale_input, ab_input, save_path=None, save_name=None):
  '''Show/save rgb image from grayscale and ab channels
     Input save_path in the form {'grayscale': '/path/', 'colorized': '/path/'}'''
  plt.clf()
  # Check if grayscale/colorized dirs exist if not the save that shit duh ヽ(͡◕ ͜ʖ ͡◕)ﾉ 
  pathlib.Path(save_path['grayscale']).mkdir(parents=True, exist_ok=True) 
  pathlib.Path(save_path['colorized']).mkdir(parents=True, exist_ok=True) 

  color_image = torch.cat((grayscale_input, ab_input), 0).numpy() # combine channels
  color_image = color_image.transpose((1, 2, 0))  # rescale for matplotlib
  cv2.imwrite(f"{save_path['colorized']}CV2_{save_name}", color_image * 255) 

  color_image[:, :, 0:1] = color_image[:, :, 0:1] * 100
  
  color_image[:, :, 1:3] = color_image[:, :, 1:3] * 255 - 128   
  
  color_image = lab2rgb(color_image.astype(np.float64))
  grayscale_input = grayscale_input.squeeze().numpy()

  if save_path is not None and save_name is not None: 
    # Export images
    plt.imsave(arr=grayscale_input, fname='{}{}'.format(save_path['grayscale'], save_name), cmap='gray')
    plt.imsave(arr=color_image, fname='{}{}'.format(save_path['colorized'], save_name))
    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(f"{save_path['colorized']}CV_{save_name}", color_image * 255) 

class AverageMeter(object):
  '''A handy class from the PyTorch ImageNet tutorial''' 
  def __init__(self):
    self.reset()
  def reset(self):
    self.val, self.avg, self.sum, self.count = 0, 0, 0, 0
  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count


def train(train_loader, model, criterion, optimizer, epoch, device, brk):
  import torch
  use_gpu = torch.cuda.is_available()
  print(f'Starting training epoch {epoch} and using GPU: {use_gpu}')
  model.train()
  
  # Prepare value counters and timers
  batch_time, data_time, losses = AverageMeter(), AverageMeter(), AverageMeter()

  end = time.time()
  for i, (target, input_gray, input_ab) in enumerate(train_loader):
    
    # Use GPU if available
    target = target.to(device)
    input_gray = input_gray.to(device)
    input_ab = input_ab.to(device)   


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
