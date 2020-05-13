import math

import numpy as np
import snrTool as snr
from matplotlib import pyplot as plt
import cv2
import pandas as pd

def main():

    imgDir = snr.loadFileDir("original image")
    # Create save directory folder to deposit diagnostic analysis
    saveDir = snr.loadFolderDir("save directory")
    saveFolder = snr.createFolder(saveDir, "Ferrites Analysis")

    # increasing contrast ...
    normal = cv2.imread(imgDir, 1)
    # Converting image to LAB color model
    lab = cv2.cvtColor(normal, cv2.COLOR_BGR2LAB)
    #snr.printImage("LAB converted", lab)

    #-----Splitting the LAB image to different channels-------------------------
    l, a, b = cv2.split(lab)
    #cv2.imshow('l_channel', l)
    #cv2.imshow('a_channel', a)
    #cv2.imshow('b_channel', b)

    #-----Applying CLAHE to L-channel-------------------------------------------
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    # cv2.imshow('CLAHE output', cl)

    #-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
    limg = cv2.merge((cl,a,b))
    #cv2.imshow('limg', limg)

    #-----Converting image from LAB Color model to RGB model--------------------
    contrast = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    # Save contrast image for diagnostic reference
    snr.saveImage(contrast, saveFolder, "Contrast-for-Filtering")
    #cv2.imshow('final', final)
    # pearlites = snr.loadFileDir("Pearlites segmented")
    # pearlites = cv2.imread(pearlites, 0)
    # ret, pearlites = cv2.threshold(pearlites, 150, 255, cv2.THRESH_BINARY)

    # Optimal numbers for 400x
    # AREA_THRESHOLD = 1000
    # FERRITE_BRIGHTNESS_THRESHOLD = 100

    AREA_THRESHOLD = 1000
    FERRITE_BRIGHTNESS_THRESHOLD = 180

    img = cv2.imread(imgDir, 0)

    #?
    #todo
    #!

    #! snr.printImage("", img)
    # img[pearlites<150] = 255
    # snr.printImage("minus pearlites", img)
    # # # apply blurring and bilateral filtering
    # img = cv2.GaussianBlur(img, (5,5), 0)
    # snr.printImage("Gaussian Blur", img)
    # img = cv2.bilateralFilter(img, 9, 120, 120)
    # snr.printImage("bilateral filtered", img)

    thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 1)
    snr.saveImage(thresh, saveFolder, "Adaptive Threshold")
    #! snr.printImage("Thresholded", thresh)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    # Draw contours
    all_contours = cv2.imread(imgDir, 1)
    cv2.drawContours(all_contours, contours, -1, (0, 0, 255), 1)
    snr.saveImage(all_contours, saveFolder, "All Contours")
    # cv2.imshow("all contours", all_contours)


    ferrite_contours = []
    areas = []
    diameters = []
    lengths = []
    widths = []
    shape_factors = []
    total_ferrite_area = 0
    # Filter based on size, as well as average intensities around centroid
    for contour in contours:

        area = cv2.contourArea(contour)

        if area < AREA_THRESHOLD:
            continue
        else:
            M = cv2.moments(contour)

            try:
                curr_cx = int(M['m10']/M['m00'])
                curr_cy = int(M['m01']/M['m00'])
                curr_centroid = (curr_cy, curr_cx)

            except ZeroDivisionError:
                print("Warning: Error in calculating moment of a segmented grain.")
                continue

            # Check a 3x3 kernel around the centroid location

            average_centroid_kernel = ( np.sum( contrast[ (curr_centroid[0]-27//2):(curr_centroid[0]+27//2 + 1),  (curr_centroid[1]-27//2):(curr_centroid[1]+27//2 + 1)]) )/ (27**2)

            if average_centroid_kernel > FERRITE_BRIGHTNESS_THRESHOLD:
                # ferrite grain has been selected, now to process data
                ferrite_contours.append(contour)
                diameters.append( math.sqrt((4*area)/math.pi) )
                areas.append(area)

                # Calculate shape factors
                left_boundary = tuple(contour[contour[:,:,0].argmin()][0])
                right_boundary = tuple(contour[contour[:,:,0].argmax()][0])
                top_boundary = tuple(contour[contour[:,:,1].argmin()][0])
                bottom_boundary = tuple(contour[contour[:,:,1].argmax()][0])

                # Get grain length and width...in PIXELS
                grain_length = (np.abs(left_boundary[0]-right_boundary[0]))
                grain_width = (np.abs(top_boundary[1]-bottom_boundary[1]))

                shape_factor = (grain_length/grain_width)
                shape_factors.append(shape_factor)
                lengths.append(grain_length)
                widths.append(grain_width)
                total_ferrite_area += area


                # debugging section
                # Draw each valid contour onto a new generated mask
                # curr_mask = cv2.imread(imgDir, 1)
                # cv2.drawContours(curr_mask, [contour], -1, (0, 0, 255), 2)
                # snr.printImage("Current Contour", curr_mask)

    # Create pandas dataframe and export it to CSV
    # Create pandas Dataframe and export it to csv
    data = {"Area (pixels^2)": areas,
            "Circular Diameter (pixels)": diameters,
            "Length (pixels)": lengths,
            "Widths (pixels)": widths,
            "Shape Factors ()": shape_factors}

    df = pd.DataFrame(data, columns=["Area (pixels^2)", "Circular Diameter (pixels)", "Length (pixels)", "Widths (pixels)", "Shape Factors ()"])

    print(df.head())
    df.to_csv("{}/data.csv".format(saveFolder), index=False)

    avg_area_pixels_squared = sum(areas)/len(areas)
    avg_grain_diameter = math.sqrt((4*avg_area_pixels_squared)/math.pi)
    avg_grain_shape_factor = sum(shape_factors)/len(areas)
    avg_grain_lengths = sum(lengths)/len(areas)
    avg_grain_widths = sum(areas)/len(areas)

    # Write text data
    txt = open("{}\\ferrites_bulk_data.txt".format(saveFolder), 'w')
    # Write analysis data into textfile:
    txtLines = []
    txtLines.append("Average Area (pixels^2): {}\n".format(avg_area_pixels_squared))
    txtLines.append("Average Grain Diameter (pixels): {}\n".format(avg_grain_diameter))
    txtLines.append("Average Grain Shape Factor (): {}\n".format(avg_grain_shape_factor))
    txtLines.append("Average Grain Length (pixels): {}\n".format(avg_grain_lengths))
    txtLines.append("Average Grain Width (pixels): {}\n".format(avg_grain_widths))

    txt.writelines(txtLines)
    txt.close()

    print(len(ferrite_contours))
    # Draw ferrite contours
    ferrite_overlay = cv2.imread(imgDir, 1)
    cv2.drawContours(ferrite_overlay, ferrite_contours, -1, (0, 0, 255), 1)
    snr.saveImage(ferrite_overlay, saveFolder, "Ferrite Outlines")
    # cv2.imshow("Ferrite Overlay outlines", ferrite_overlay)


    filled_ferrites_overlay = cv2.imread(imgDir, 1)
    cv2.drawContours(filled_ferrites_overlay, ferrite_contours, -1, (255, 255, 0), -1)
    # cv2.imshow("ferrite contours", filled_ferrites_overlay)
    snr.saveImage(filled_ferrites_overlay, saveFolder, "Ferrites Filled")

    dst = cv2.addWeighted(ferrite_overlay, 0.7, filled_ferrites_overlay, 0.3, 0)
    # snr.printImage("weighted", dst)
    snr.saveImage(dst, saveFolder, "Ferrites-Boundaries-Mixed")

    # Draw ferrites mask
    ferrites_mask = np.full_like(filled_ferrites_overlay, (255, 255, 255), dtype=np.uint8)
    cv2.drawContours(ferrites_mask, ferrite_contours, -1, (0, 255, 0), -1)
    snr.saveImage(ferrites_mask, saveDir, "Final Ferrites Mask")


    # Draw histogram
    num_bins = 25
    n, bins, patches = plt.hist(areas, num_bins, facecolor='blue', alpha=0.6)
    plt.xlabel("Grain Area (pixles^2)")
    plt.ylabel("Grains Count ()")
    plt.title("Histogram of Ferrite Grain Size Distribution")

    plt.savefig("{}\\ferrite-distribution.png".format(saveFolder))
main()