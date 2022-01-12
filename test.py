import cv2    
import sys
import os


def plotData():
    img1 = cv2.imread('dummyData/ColorfulOriginal/Apple/Apple1.jpg')
    img2 = cv2.imread('dummyData/ColorfulOriginal/Apple/Apple3.jpg')

    img1[img1[:, :, 1:].all(axis=-1)] = 0
    img2[img2[:, :, 1:].all(axis=-1)] = 0

    dst = cv2.addWeighted(img1, 0.5, img2, 0.7, 0)

    cv2.imshow('sas', dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def showSubtracted(img1, img2):
    e = os.path.exists(img1)
    e = os.path.exists(img2)
    img1cv = cv2.imread(img1)
    img2cv = cv2.imread(img2)

    difference = cv2.subtract(img1cv, img2cv)
    cv2.imshow("Original", difference)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    arg1 = None
    arg2 = None
    if(len(sys.argv) > 0):
        arg1 = os.path.abspath(sys.argv[1])
        arg2 = os.path.abspath(sys.argv[2])
    showSubtracted(arg1, arg2)

if __name__ == '__main__':
    main()