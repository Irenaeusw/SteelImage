import numpy as np
import snrTool as snr
from matplotlib import pyplot as plt
import cv2

imgDir = snr.loadFileDir("original image")
pearlites = snr.loadFileDir("Pearlites segmented")
pearlites = cv2.imread(pearlites, 0) 
ret, pearlites = cv2.threshold(pearlites, 150, 255, cv2.THRESH_BINARY) 


img = cv2.imread(imgDir, 0)

snr.printImage("", img)
img[pearlites<150] = 255
snr.printImage("minus pearlites", img) 
# apply blurring and bilateral filtering
img = cv2.GaussianBlur(img, (5,5), 0)
snr.printImage("Gaussian Blur", img)
img = cv2.bilateralFilter(img, 9, 120, 120)
snr.printImage("bilateral filtered", img) 

thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
snr.printImage("Thresholded", thresh)

#todo Find contours of the image, and use same technique used in DeltaHacks VI to filter each contour based on the grey value in a kernel from the 
#todo centroid calculated from each contour. 