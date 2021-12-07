import cv2 as cv
import numpy as np 

# rescale function
def rescaleFrame(frame, scale=0.75):
    height = int(frame.shape[0]*scale)
    width = int(frame.shape[1]*scale)
    dimension = (width, height)
    return cv.resize(frame, dimension, interpolation= cv.INTER_AREA)


# rescale image
img = cv.imread('sapiens.jpg')

img = rescaleFrame(img, scale = 2)


gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
sift = cv.SIFT_create()
kp = sift.detect(gray,None)

kp, des = sift.detectAndCompute(gray,None)


# draw without size of SIFT

# img=cv.drawKeypoints(gray,kp,img)
# cv.imshow('sift_keypoints.jpg',img)

img=cv.drawKeypoints(gray,kp,img,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv.imshow('sift_keypoints.jpg',img)

cv.waitKey()





