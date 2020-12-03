# Import ML, image and graphing libraries
import os
import re
import numpy as np
import argparse
import imutils
import cv2
from natsort import natsorted, ns
from skimage import io
from sklearn import svm, metrics
import matplotlib.pyplot as plt
# Import pre-processing class
# from anpr_pre_processing import LocateAndDivideNumberPlate

# Import and process training and test data (and labels for training)
def import_data(args):
    train_list = []
    train_labels = []
    test_list = []
    test_labels = []

    train_dir = args["input"] + '/train_characters/'
    test_dir = args["input"] + '/test_characters/'

    train_files = natsorted(os.listdir(train_dir))
    not_folder = re.compile('^.*\.(py)$')
    train_files = [i for i in train_files if not not_folder.match(i)]
    test_files = natsorted(os.listdir(test_dir))

    for im_folder in train_files:
        char_dir = train_dir + im_folder + '/'
        char_images = natsorted(os.listdir(char_dir))
        # Get training labels and images
        for label in char_images:
            train_labels.append(label[0])
            train_list.append(cv2.resize(cv2.imread(char_dir + label, 0), (35,60)))

    # Get training labels
    for label in test_files:
        test_labels.append(label[0]) # S or 5

    # Add character images to the test list
    for c in test_files:
        test_list.append(cv2.resize(cv2.imread(test_dir + c, 0), (35,60)))

    return (train_list, train_labels, test_list, test_labels)

# Create Support Vector Machine classifier
def build_SVM(train_data, train_labels):
    # Instantiate class and train on data
    svc = svm.SVC(kernel='linear', C=1).fit(train_data, train_labels) # Default C and gamma
    return svc

def main():
    # Image data folder
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True)
    args = vars(ap.parse_args())

    print("Importing training and test data...")
    (characters, labels, test, test_labels) = import_data(args)
    flatten_chars = np.reshape(characters, (len(characters), 35*60))
    flatten_test = np.reshape(test, (len(test), 35*60))

    # for c in characters:
    #     cv2.imshow('Character', c)
    #     cv2.waitKey(0)

    print("Building & training SVM classifier...")
    svc = build_SVM(flatten_chars, labels);

    print("\t\t---------Validating training data---------")
    pred = svc.predict(flatten_chars)
    training_accuracy = 0
    for idx, p in enumerate(pred):
        if p == labels[idx]:
            training_accuracy += 1
    training_accuracy /= len(pred)
    training_accuracy *= 100
    print("Training Accuracy: " + str(training_accuracy) + "%")

    print("\n\t\t---------Predicting test data---------")
    pred = svc.predict(flatten_test)
    print("Predicted:\n" + str(list(pred)))
    print("Actual:\n" + str(test_labels))
    test_accuracy = 0
    for idx, p in enumerate(pred):
        if p == test_labels[idx]:
            test_accuracy += 1
    test_accuracy /= len(pred)
    test_accuracy *= 100
    print("Testing Accuracy: " + str(test_accuracy) + "%")

main() # Start ANPR
