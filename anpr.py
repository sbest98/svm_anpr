# Import ML, image and graphing libraries
import os
import re
import numpy as np
import argparse
import imutils
import cv2
import math
from joblib import load
from natsort import natsorted, ns
from sklearn import svm, metrics
import matplotlib.pyplot as plt
# Import pre-processing class
from anpr_pre_processing import extract_svm_characters

ap = argparse.ArgumentParser()
ap.add_argument("-i", required=True)
ap.add_argument("-d")
arg = vars(ap.parse_args())
plate_found, image_list, position_list = extract_svm_characters(arg["i"],
                                                                int(arg["d"])
                                                                if ("d" in arg)
                                                                else False)


if plate_found:
    sorted_image_list = []
    sorted_position_list = np.array(position_list)[:,0].tolist()
    num_chars = len(sorted_position_list)
    # Sort by x position in image
    for idx, pos in enumerate(sorted_position_list): sorted_position_list[idx] = [idx, pos]
    sorted_position_list.sort(key=lambda x: x[1])
    # Add distance (pixels) between characters (excluding last)
    for idx, pos in enumerate(sorted_position_list[:num_chars-1]):
        pos.append(sorted_position_list[idx+1][1] - (pos[1] + position_list[pos[0]][2]) )
    # Determine mean and std
    mean = 0
    std = 0
    for img in sorted_position_list[:num_chars-1]:
        mean += img[2]
    mean /= (num_chars-1)
    for img in sorted_position_list[:num_chars-1]:
        std += math.pow(img[2]-mean, 2)
    std = math.sqrt(std/(num_chars-2))
    # Ideal character distance range
    upper_d = mean + std
    lower_d = mean - std
    print(sorted_position_list)
    print('Mean(u): ' + str(mean),'Std: ' + str(std))
    print('u+std: ' + str(upper_d),'u-std: ' + str(lower_d))

    print("Importing ANPR SVM Classifier...")
    svc = load("./trained_models/ANPR_SVM_v1.joblib")

    image_list = np.reshape(image_list, (len(image_list), 35*60))
    plate_predictions = svc.predict(image_list)
    sorted_plate_predictions = []
    for pos in sorted_position_list:
        sorted_plate_predictions.append(plate_predictions[pos[0]])
    plate_string = ''.join([str(elem) for elem in sorted_plate_predictions])
    print("Number plate: " + plate_string)
else:
    print("Plate not found!")
