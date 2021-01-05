# Import ML, image and graphing libraries
import os
import re
import numpy as np
import argparse
import imutils
import cv2
from joblib import load
from natsort import natsorted, ns
from sklearn import svm, metrics
import matplotlib.pyplot as plt
# Import pre-processing class
from anpr_pre_processing import extract_svm_characters

ap = argparse.ArgumentParser()
ap.add_argument("-i", required=True)
arg = vars(ap.parse_args())
plate_found, image_list, position_list = extract_svm_characters(arg["i"])


if plate_found:
    sorted_image_list = []
    print(position_list)
    sorted_position_list = np.array(position_list)[:,0].tolist()
    for idx, pos in enumerate(sorted_position_list): sorted_position_list[idx] = [idx, pos]
    sorted_position_list.sort(key=lambda x: x[1])

    print("Importing ANPR SVM Classifier...")
    svc = load("./trained_models/ANPR_SVM_v1.joblib")

    image_list = np.reshape(image_list, (len(image_list), 35*60))
    plate_predictions = svc.predict(image_list)
    sorted_plate_predictions = []
    for pos in sorted_position_list:
        sorted_plate_predictions.append(plate_predictions[pos[0]])
    plate_string = ''.join([str(elem) for elem in sorted_plate_predictions])
    print("Number plate: " + plate_string)
