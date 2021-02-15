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

def getANPR(image_list, position_list, d):
    sorted_image_list = []
    sorted_position_list = np.array(position_list)[:,0].tolist()
    num_chars = len(sorted_position_list)
    # Sort by x position in image
    for idx, pos in enumerate(sorted_position_list): sorted_position_list[idx] = [idx, pos]
    sorted_position_list.sort(key=lambda x: x[1])
    # Add distance (pixels) between characters
    #   - [Left, Right]
    for idx, pos in enumerate(sorted_position_list):
        if idx == 0 :
            pos.append(0.0) # Nothing to left
            pos.append(sorted_position_list[idx+1][1] - (pos[1] + position_list[pos[0]][2]) )
        elif idx == num_chars-1:
            pos.append(pos[1] - (sorted_position_list[idx-1][1] + position_list[sorted_position_list[idx-1][0]][2]) )
            pos.append(0.0) # Nothing to right
        else:
            pos.append(pos[1] - (sorted_position_list[idx-1][1] + position_list[sorted_position_list[idx-1][0]][2]) )
            pos.append(sorted_position_list[idx+1][1] - (pos[1] + position_list[pos[0]][2]) )

    # Determine mean and std
    #   - Right side gaps
    mean_r = 0
    std_r = 0
    for img in sorted_position_list[:num_chars-1]:
        mean_r += img[2]
    mean_r /= (num_chars-1)
    for img in sorted_position_list[:num_chars-1]:
        std_r += math.pow(img[2]-mean_r, 2)
    std_r = math.sqrt(std_r/(num_chars-2))
    # Ideal character distance range
    upper_d_r = mean_r + (std_r)
    lower_d_r = mean_r - (std_r)
    if d == True:
        print(sorted_position_list)
        print("Gaps between characters:")
        print('\tMean(u): ' + str(mean_r),'Std: ' + str(std_r))
        print('\tu+0std: ' + str(upper_d_r),'u-std: ' + str(lower_d_r))

    # Determine center gap of plate and noise characters
    d_char = [] # Dirty bit characters
    char_counter = 0
    centre_index = False
    for idx, char in enumerate(sorted_position_list):
        if idx == 0:
            if (char[3] >= lower_d_r) and (char[3] <= upper_d_r):
                d_char.append(0)
                char_counter += 1
            else:
                d_char.append(1)
        elif idx == num_chars-1:
            if (char[2] >= lower_d_r) and (char[2] <= upper_d_r):
                d_char.append(0)
                char_counter += 1
            else:
                d_char.append(1)
                if d_char[idx-1] == 1:
                    d_char[idx-1] = 0
        else:
            if ((char[3] >= lower_d_r) and (char[3] <= upper_d_r)) and ((char[2] >= lower_d_r) and (char[2] <= upper_d_r)):
                d_char.append(0)
                char_counter += 1
            else:
                if d_char[idx-1] == 1:
                    d_char.append(0)
                    char_counter += 1
                elif char_counter == 3:
                    d_char.append(0)
                    char_counter = 0
                    centre_index = char[0]
                elif centre_index == sorted_position_list[idx-1][0]:
                    d_char.append(0)
                else:
                    d_char.append(1)
    if d: print(d_char)

    print("Importing ANPR SVM Classifier...")
    svc = load("./trained_models/ANPR_SVM_v1.joblib")

    image_list = np.reshape(image_list, (len(image_list), 35*60))
    plate_predictions = svc.predict(image_list)
    sorted_plate_predictions = []
    for idx, pos in enumerate(sorted_position_list):
        if not d_char[idx]:
            sorted_plate_predictions.append(plate_predictions[pos[0]])
    plate_string = ''.join([str(elem) for elem in sorted_plate_predictions])
    if len(plate_string) == 7:
        print("Number plate:\n" + plate_string)
        return 2
    elif len(plate_string) > 7:
        if d: print("Debug plate value: ", plate_string)
        return 1
    else:
        return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", required=True)
    ap.add_argument("-d")
    arg = vars(ap.parse_args())
    plate_found, image_list, position_list = extract_svm_characters(arg["i"],
                                                                    int(arg["d"])
                                                                    if ("d" in arg)
                                                                    else False)

    if plate_found:
        anpr_status = getANPR(image_list, position_list, int(arg["d"]))
        if anpr_status == 1:
            print("Noise detected. Attempting border removal...")
            plate_found, image_list, position_list = extract_svm_characters(arg["i"],
                                                                            int(arg["d"])
                                                                            if ("d" in arg)
                                                                            else False,
                                                                            True)
            if plate_found:
                anpr_status = getANPR(image_list, position_list, int(arg["d"]))
                if not anpr_status:
                    print("Plate not classified!")
            else:
                print("Plate not classified!")
        elif anpr_status == 0:
            print("Plate not classified!")
    else:
        print("Plate not classified!")


if __name__ == '__main__':
    main()
