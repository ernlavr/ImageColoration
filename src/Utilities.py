import numpy as np
import torch
import cv2

def showSubtracted(img1, img2):
    difference = cv2.subtract(img1, img2)
    cv2.imshow("Original", difference)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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

def convertTensorToCV(img):
    imgConverted :np.ndarray = np.moveaxis(img.numpy()*255, 0, -1).astype(np.uint8)
    imgConverted = imgConverted[:, :, :, 0]
    return imgConverted.transpose((1, 2, 0))


def showTensor(L, AB):
    # lConv = L.numpy()
    # lConv = lConv.transpose((1, 2, 0))
    max = torch.max(AB)
    min = torch.min(AB)
    

    lScaled = convertTensorToCV(L)
    abScaled = convertTensorToCV(AB)

    arr1 = abScaled[:, :, 0]
    arr2 = abScaled[:, :, 1]
    
    result = (arr1 == arr2).all()
    result = (arr1 == lScaled).all()
    result = (arr2 == lScaled).all()

    showImgLab(lScaled, abScaled)
    

def showImgLab(L, AB):
    merged = np.dstack((L, AB)) 
    RGB = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    R = RGB[:, :, 0]
    G = RGB[:, :, 1]
    B = RGB[:, :, 2]
    RGB = np.asarray([R, G, B], dtype=np.uint8)
    RGB = RGB.transpose((1, 2, 0))
    showImg(RGB)

def showImg(RGB):
    cv2.imshow('Img', RGB)
    cv2.waitKey(0)
    cv2.destroyAllWindows()