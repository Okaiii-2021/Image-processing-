# rescale function
def rescaleFrame(frame, scale=0.75):
    height = int(frame.shape[0]*scale)
    width = int(frame.shape[1]*scale)
    dimension = (width, height)
    return cv.resize(frame, dimension, interpolation= cv.INTER_AREA)