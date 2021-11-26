import cv2    

img1 = cv2.imread('dummyData/ColorfulOriginal/Apple/Apple1.jpg')
img2 = cv2.imread('dummyData/ColorfulOriginal/Apple/Apple3.jpg')

img1[img1[:, :, 1:].all(axis=-1)] = 0
img2[img2[:, :, 1:].all(axis=-1)] = 0

dst = cv2.addWeighted(img1, 0.5, img2, 0.7, 0)

cv2.imshow('sas', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()