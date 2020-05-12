import numpy as np
import cv2
from matplotlib import pyplot as plt
import snrTool as snr

#! alsdkjfsl asdfdfdf dfdf
#? Commit i think?  making more comments weee
#! Commit test 2 asf sdfsdf asdfasdf dstestubg 2 reeeeee

def build_filters(kernel_size, sigma):
    filters = []
    for theta in np.arange(0, np.pi, np.pi / 16):
        kern = cv2.getGaborKernel(kernel_size, sigma, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        kern /= 1.5*kern.sum()
        filters.append(kern)
        return filters

def process_gabor_accumulated(img, gabor_filter):
    """[Applies various gabor_filters to analyze structural and directional data of an input image.]

    Arguments:
        img {[np.array(2D)]} -- [input image]
        gabor_filter {[np.array(2D)]} -- [input gabor_filters]

    Returns:
        [type] -- [list of accumulated images]
    """
    accumulated = np.zeros_like(img)
    for kernel in gabor_filter:
        curr_img = cv2.filter2D(img, cv2.CV_8UC3, kernel)
        np.maximum(accumulated, curr_img, accumulated)
    return accumulated


def findPearliteBands(imgDir, magnification):
    img = cv2.imread(imgDir, 0)
    if magnification == 400:
        threshold =  220
    elif magnification == 100:
        threshold = 200

    # process with an orientation of 0
    kernel_0 = cv2.getGaborKernel((9,9), sigma=4.0, theta=np.pi/2, lambd=2.7, gamma=0.25,ktype=cv2.CV_32F)
    processed_0 = cv2.filter2D(img, cv2.CV_8UC3, kernel_0)
    # snr.printImage("0 degrees orientated gabor filter applied", processed_0)

    line_mask = cv2.imread(imgDir, 1)
    # Apply hough transform to look for horizontal lines
    # dilated = cv2.dilate(processed_0, (9,9), iterations=1)
    # snr.printImage("", dilated)
    edges = cv2.Canny(processed_0, 50,150,apertureSize = 3)
    # snr.printImage("", edges)
    # snr.printImage("edge map of horizontal processed lines", edges)
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=threshold)
    print(len(lines))
    for line in lines:
        rho = line[0,0]
        theta = line[0,1]

        # check to see if rho is relatively straight...
        # between 80 to 100 degrees
        if theta < 1.39626 or theta > 1.74533:
            continue
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 2000*(-b))
        y1 = int(y0 + 2000*(a))
        x2 = int(x0 - 2000*(-b))
        y2 = int(y0 - 2000*(a))

        cv2.line(line_mask,(x1,y1),(x2,y2),(0,0,255),1)

    snr.printImage("Lines detected overlayed", line_mask)

    img_colour = cv2.imread(imgDir, 1)
    img_colour[np.argwhere(processed_0>5)[:,0], np.argwhere(processed_0>5)[:,1]] = (0,165, 255)
    # snr.printImage("horizontal detected overlayed", img_colour)

imgDir = snr.loadFileDir("Image Directory")
findPearliteBands(imgDir, 400)