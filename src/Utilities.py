import cv2
import numpy as np
import torch

def getStacked(L, AB):
    merged = np.dstack((L, AB)) 
    RGB = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return RGB

def getTensConverted(L, AB):
    lScaled = np.moveaxis(L.numpy()*255, 0, -1).astype(np.uint8)
    abScaled = np.moveaxis(AB.numpy()*255, 0, -1).astype(np.uint8)
    return getStacked(lScaled, abScaled)

def compareTwoTensors(t1, t2):
    t1 = np.moveaxis(t1.numpy()*255, 0, -1).astype(np.uint8)
    t2 = np.moveaxis(t2.numpy()*255, 0, -1).astype(np.uint8)

    t1 = t1[:, :, :, 0]
    t1 = t1.transpose((1, 2, 0))
    t2 = t2[:, :, :, 0]
    t2 = t2.transpose((1, 2, 0))

    print(t2 - t1)


def showTensor(L, AB):
    # lConv = L.numpy()
    # lConv = lConv.transpose((1, 2, 0))
    max = torch.max(AB)
    min = torch.min(AB)

    lScaled :np.ndarray = np.moveaxis(L.numpy()*255, 0, -1).astype(np.uint8)
    abNumpy = AB.numpy() * 255
    abMoved = np.moveaxis(abNumpy, 0, -1)
    abInt = abMoved.astype(np.uint8)
    

    lScaled = lScaled[:, :, :, 0]
    lScaled = lScaled.transpose((1, 2, 0))
    abScaled = abInt[:, :, :, 0]
    abScaled = abScaled.transpose((1, 2, 0))

    showImgLab(lScaled, abScaled)
    

def showImgLab(L, AB):
    merged = np.dstack((L, AB)) 
    RGB = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    showImg(RGB)

def showImg(RGB):
    cv2.imshow('Img', RGB)
    cv2.waitKey(0)
    cv2.destroyAllWindows()