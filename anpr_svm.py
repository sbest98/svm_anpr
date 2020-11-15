# Import ML, image and graphing libraries
import os
import numpy as np
import argparse
import imutils
import cv2
from natsort import natsorted, ns
from skimage import io
from sklearn import svm, metrics
import matplotlib.pyplot as plt

# Import and process training and test data (and labels for training)
def import_data(args):
    train_list = []
    train_labels = []
    test_list = []
    test_labels = []
    train_files = natsorted(os.listdir(args["input"] + '/train_characters/'))
    test_files = natsorted(os.listdir(args["input"] + '/test_characters/'))
    # Get training labels
    for label in train_files:
        train_labels.append(label[0]) # S or 5
    # Get training labels
    for label in test_files:
        test_labels.append(label[0]) # S or 5
    # Add character images to the train list
    for c in train_files:
        train_list.append(cv2.resize(cv2.imread(args["input"] + '/train_characters/' + c, 0), (35,60)))
    # Add character images to the test list
    for c in test_files:
        test_list.append(cv2.resize(cv2.imread(args["input"] + '/test_characters/' + c, 0), (35,60)))
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
    flatten_chars = np.reshape(characters, (20, 35*60))
    flatten_test = np.reshape(test, (len(test), 35*60))
    # for c in characters:
    #     cv2.imshow('Character', c)
    #     cv2.waitKey(0)
    print("Building & training SVM classifier...")
    svc = build_SVM(flatten_chars, labels);
    print("\t\t---------Validating training data---------")
    pred = svc.predict(flatten_chars)
    print("Predicted:\n" + str(list(pred)))
    print("Actual:\n" + str(labels))
    print("\n\t\t---------Predicting test data---------")
    pred = svc.predict(flatten_test)
    print("Predicted:\n" + str(list(pred)))
    print("Actual:\n" + str(test_labels))

main() # Start ANPR
