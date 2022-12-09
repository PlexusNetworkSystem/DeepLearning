import numpy as np
import cv2 as cv

# border widht of detected face
shiftValue = 20

# camera size
resizeX = 460
resizeY = 300

# succession ratio limit of detection
threshold = 0.3

# read model files and create DNN
#dnnNetwork = cv.dnn.readNetFromCaffe("data/deploy.prototxt.txt", "data/res10SSD.caffemodel")

# Start camera
videoCapture = cv.VideoCapture(0)
