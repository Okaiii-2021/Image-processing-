import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def rescaleFrame(frame, scale=0.75):
    height = int(frame.shape[0]*scale)
    width = int(frame.shape[1]*scale)
    dimension = (width, height)
    return cv.resize(frame, dimension, interpolation= cv.INTER_AREA)





img = cv.imread("chess.png")
img = rescaleFrame(img, scale=0.5)

gray = cv.imread("chess.png", 0)
gray = rescaleFrame(gray, scale=0.5)


gray = np.float32(gray)
dst = cv.cornerHarris(gray,2,3,0.04)

print(gray.shape) 
print(dst.shape) # => ket qua tra ve la matrix voi gia tri R tren tung pixel

#result is dilated for marking the corners, not important
dst = cv.dilate(dst,None)

# Threshold for an optimal value, it may vary depending on the image.
img[dst>0.01*dst.max()]=[0,0,255]

cv.imshow('dst',img)
if cv.waitKey(0) & 0xff == 27:
    cv.destroyAllWindows()