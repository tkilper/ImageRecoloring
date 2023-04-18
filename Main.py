# CS 440 Project 4
# Tristan Kilper (twk28)

import sys
sys.path.append('.')
import numpy as np
import cv2
from BasicAgent import BasicAgent as ba
from ImprovedAgent import ImprovedAgent as ia

"""
# load original image using OpenCV, convert to black and white, and save it in new file
orgimage = cv2.imread("streetindamascus.jpg")
bwimage = cv2.cvtColor(orgimage, cv2.COLOR_BGR2GRAY)
cv2.imwrite("sidgray.jpg",bwimage)
"""

# load original color and gray image
orgimage = cv2.imread("streetindamascus.jpg")
bwimage = cv2.imread("sidgray.jpg")

# ***basic Agent Testing***
# *************************
# find 5 representative colors
repcolors = ba.findRepColors(orgimage, 5)
print(repcolors)

#repcolors = [[72.13381796, 117.45799791, 143.72990376],[252.88004491, 254.21646611, 254.60008508],[29.82149206, 61.45775299, 71.85256606],[180.77784955, 174.43226612, 136.67419407],[133.65888511, 182.93359569, 212.56930424]]

# recolor the image
image, righttest, lefttest = ba.recolor(orgimage, bwimage, repcolors)
cv2.imwrite("righttest.jpg", righttest)
cv2.imwrite("lefttest.jpg", lefttest)

# ***improved agent testing***
# ****************************
# obtain model parameters for each color 
rw = ia.trainNetwork(ia, orgimage, 0.000001, 'red')
bw = ia.trainNetwork(ia, orgimage, 0.000001, 'blue')
gw = ia.trainNetwork(ia, orgimage, 0.000001, 'green')

# recolor the image
iarighttest = ia.recolor(orgimage, rw, bw, gw)
cv2.imwrite('iarighttest.jpg', iarighttest)