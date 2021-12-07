import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def rescaleFrame(frame, scale=0.75):
    height = int(frame.shape[0]*scale)
    width = int(frame.shape[1]*scale)
    dimension = (width, height)
    return cv.resize(frame, dimension, interpolation= cv.INTER_AREA)


img1 = cv.imread('run1.jpg',cv.IMREAD_GRAYSCALE)          # queryImage
img1 = rescaleFrame(img1, scale=2)



img2 = cv.imread('run2.jpg',cv.IMREAD_GRAYSCALE) # trainImage
img2 = rescaleFrame(img2, scale=0.5)


# Initiate SIFT detector
sift = cv.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
# BFMatcher with default params
bf = cv.BFMatcher()
matches = bf.knnMatch(des1,des2,k=2)

print(len(matches))
print(type(matches))
print(matches)

# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])
# cv.drawMatchesKnn expects list of lists as matches.
img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(img3),plt.show()