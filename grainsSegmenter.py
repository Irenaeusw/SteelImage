import cv2
import snrTool as snr
import numpy as np
from matplotlib import pyplot as plt
import math
import pandas as pd

imgDir = snr.loadFileDir("img directory")
saveDir = snr.loadFolderDir("save directory")

saveFolder = snr.createFolder(saveDir, "Processing Steps Results")

txt = open("{}\\pearlites_data.txt".format(saveFolder), 'w')

#load image into colour
img = cv2.imread(imgDir, 1)
img_2 = cv2.imread(imgDir, 1)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
height, width = img_gray.shape
# saveDir = snr.loadFolderDir("save directory")

# Pre-Process images ------------#
# Thresholding
# thresh = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 1)

# Otsu thresholding after gaussian filtering
blur = cv2.GaussianBlur(img_gray, (5,5), 0)
snr.saveImage(blur, saveFolder, "Gaussian Blurred 5x5")
ret3, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
snr.saveImage(thresh, saveFolder, "Thresholded Otsu")
# snr.printImage("Thresholded", thresh)

# Erode the image
kernel = np.ones( (3,3), np.uint8)
erosion = cv2.erode(thresh, kernel, iterations=1)
erosion_colour = cv2.cvtColor(erosion, cv2.COLOR_GRAY2BGR)
snr.saveImage(erosion_colour, saveFolder, "Erosion Kernel 3x3")
# snr.printImage("Eroded", erosion)

contours, hierarchy = cv2.findContours(erosion, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
# Draw all original contours
cv2.drawContours(img_2, contours, -1, (0, 255, 0), 2)
snr.saveImage(img_2, saveFolder, "All Contours-No Size Discrimination")

# Filter contours
pearlites = []
areas = []
sum_shape_factor = 0
sum_lengths = 0
sum_widths = 0
lengths = []
widths = []
shape_factors = []
diameters = []

for contour in contours:

    curr_area = cv2.contourArea(contour)
    if curr_area > 20 and curr_area < 900000:
        pearlites.append(contour)
        areas.append(curr_area)

        # Calculate circular diameters of the grain in PIXELS
        diameters.append( math.sqrt((4*curr_area)/math.pi) )

        # Calculate Length and Width of the current contour
        left_boundary = tuple(contour[contour[:,:,0].argmin()][0])
        right_boundary = tuple(contour[contour[:,:,0].argmax()][0])
        top_boundary = tuple(contour[contour[:,:,1].argmin()][0])
        bottom_boundary = tuple(contour[contour[:,:,1].argmax()][0])

        # Get grain length and width...in PIXELS
        grain_length = (np.abs(left_boundary[0]-right_boundary[0]))
        grain_width = (np.abs(top_boundary[1]-bottom_boundary[1]))

        shape_factor = (grain_length/grain_width)
        sum_shape_factor += shape_factor
        sum_lengths += grain_length
        sum_widths += grain_width
        lengths.append(grain_length)
        widths.append(grain_width)
        shape_factors.append(shape_factor)


# Create pandas Dataframe and export it to csv
data = {"Area (pixels^2)": areas,
        "Circular Diameter (pixels)": diameters,
        "Length (pixels)": lengths,
        "Widths (pixels)": widths,
        "Shape Factors ()": shape_factors}

df = pd.DataFrame(data, columns= ["Area (pixels^2)", "Circular Diameter (pixels)", "Length (pixels)", "Widths (pixels)", "Shape Factors ()"])

print(df.head())
df.to_csv("{}/data.csv".format(saveFolder), index=False)

avg_area_pixels_squared = sum(areas)/len(areas)
avg_grain_diameter = math.sqrt((4*avg_area_pixels_squared)/math.pi)
avg_grain_shape_factor = sum_shape_factor/len(areas)
avg_grain_lengths = sum_lengths/len(areas)
avg_grain_widths = sum_widths/len(areas)

# Write analysis data into textfile:
txtLines = []
txtLines.append("Average Area (um^2): {}\n".format(avg_area_pixels_squared))
txtLines.append("Average Grain Diameter (um): {}\n".format(avg_grain_diameter))
txtLines.append("Average Grain Shape Factor (): {}\n".format(avg_grain_shape_factor))
txtLines.append("Average Grain Length (um): {}\n".format(avg_grain_lengths))
txtLines.append("Average Grain Width (um): {}\n".format(avg_grain_widths))

txt.writelines(txtLines)
txt.close()


cv2.drawContours(img, pearlites, -1, (0, 0, 255), 2)
cv2.drawContours(erosion_colour, pearlites, -1, (0, 0, 255), 2)

# for i in range(len(pearlites)):
#     print(cv2.contourArea(pearlites[i]))
# snr.printImage("Contours Overlayed", img)
snr.saveImage(img, saveFolder, "Pearlites Overlayed Original Image")

# Draw new pearlites strict overlay
pearlites_mask = np.full_like(img, (255, 255, 255), dtype=np.uint8)
cv2.drawContours(pearlites_mask, pearlites, -1,  (255, 0, 0), -1)
# snr.printImage("only pearlites", pearlites_mask)
snr.saveImage(pearlites_mask, saveFolder, "Final Pearlites Mask")
snr.saveImage(pearlites_mask, saveDir, "Final Pearlites Mask")

# Draw histogram of pearlite size/area distribution
num_bins = 10
n, bins, patches = plt.hist(areas, num_bins, facecolor='blue', alpha=0.6)
plt.xlabel("Grain Area (pixles^2)")
plt.ylabel("Grains Count ()")
plt.title("Histogram of Pearlite Grain Size Distribution")

plt.savefig("{}\\pearlites-distribution.png".format(saveFolder))