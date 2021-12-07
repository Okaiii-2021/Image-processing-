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



# Initiate FAST object with default values
fast = cv.FastFeatureDetector_create(threshold = 10)

fast.setNonmaxSuppression(0)

# find and draw the keypoints
kp = fast.detect(gray,None)
img2 = cv.drawKeypoints(gray, kp, None, color=(255,0,0))



# Print all default params
print( "Threshold: {}".format(fast.getThreshold()) )
print( "nonmaxSuppression:{}".format(fast.getNonmaxSuppression()) )
print( "neighborhood: {}".format(fast.getType()) )
print( "Total Keypoints with nonmaxSuppression: {}".format(len(kp)) )



cv.imshow('dst',img2)
if cv.waitKey(0) & 0xff == 27:
    cv.destroyAllWindows()