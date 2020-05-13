import math

import numpy as np
import snrTool as snr
from matplotlib import pyplot as plt
import cv2
import pandas as pd

def compareMicrostructures():

    originalImageDir = snr.loadFileDir("Original image")
    ferritesDir = snr.loadFileDir("Ferrites Overlay")
    pearlitesDir = snr.loadFileDir("Pearlites Overlay")
    saveDir = snr.loadFolderDir("save directory")

    ferrites = cv2.imread(ferritesDir, 1)
    pearlites = cv2.imread(pearlitesDir, 1)
    original_image = cv2.imread(originalImageDir, 1)

    final = cv2.addWeighted(ferrites, 0.5, pearlites, 0.5, 0)
    final = cv2.addWeighted(final, 0.4, original_image, 0.6, 0)
    snr.saveImage(final, saveDir, "Microstructure Comparison")

compareMicrostructures()