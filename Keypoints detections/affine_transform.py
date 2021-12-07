import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


img = cv.imread("car.jpg",0)

def rescaleFrame(frame, scale=0.75):
    height = int(frame.shape[0]*scale)
    width = int(frame.shape[1]*scale)
    dimension = (width, height)
    return cv.resize(frame, dimension, interpolation= cv.INTER_AREA)

img = rescaleFrame(img, scale=0.5)

rows,cols = img.shape[0:2]
pts1 = np.float32([[50,50],[200,50],[50,200]])
pts2 = np.float32([[10,100],[200,50],[100,250]])

M = cv.getAffineTransform(pts1,pts2)

dst = cv.warpAffine(img,M,(cols,rows))

plt.subplot(121),plt.imshow(img, cmap= "gray"),plt.title('Input')
plt.subplot(122),plt.imshow(dst, cmap= "gray"),plt.title('Output')
plt.show()

