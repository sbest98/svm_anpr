# Import ML, image and graphing libraries
import os
import re
import numpy as np
import argparse
import imutils
import cv2
import random
from joblib import dump
from natsort import natsorted, ns
from skimage import io
from sklearn import svm, metrics
from sklearn.model_selection import RepeatedKFold
import matplotlib.pyplot as plt


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
    test_files = [i for i in test_files if not not_folder.match(i)]

    for im_folder in train_files:
        char_dir = train_dir + im_folder + '/'
        char_images = natsorted(os.listdir(char_dir))
        # Get training labels and images
        for label in char_images:
            train_labels.append(label[0])
            train_list.append(cv2.resize(cv2.imread(char_dir + label, 0), (35,60)))

    for im_folder in test_files:
        char_dir = test_dir + im_folder + '/'
        char_images = natsorted(os.listdir(char_dir))
        # Get training labels and images
        for label in char_images:
            test_labels.append(label[0])
            test_list.append(cv2.resize(cv2.imread(char_dir + label, 0), (35,60)))

    return (train_list, train_labels, test_list, test_labels)

def import_k_fold(args):
    X_list = []
    X_labels = []

    data_dir = args["input"] + '/k_characters/'

    char_files = natsorted(os.listdir(data_dir))
    not_folder = re.compile('^.*\.(py)$')
    char_files = [i for i in char_files if not not_folder.match(i)]

    for im_folder in char_files:
        char_dir = data_dir + im_folder + '/'
        char_images = natsorted(os.listdir(char_dir))
        # Get training labels and images
        for label in char_images:
            X_labels.append(label[0])
            X_list.append(cv2.resize(cv2.imread(char_dir + label, 0), (35,60)))

    return (X_list, np.array(X_labels))

# Create Support Vector Machine classifier
def build_SVM(train_data, train_labels):
    # Instantiate class and train on data
    svc = svm.SVC(kernel='linear', C=1).fit(train_data, train_labels) # Default C and gamma
    return svc

# K fold cross validation
def k_fold_main():
    # Image data folder
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True)
    args = vars(ap.parse_args())

    print("Importing training and test data...")
    (characters, labels) = import_k_fold(args)
    flatten_chars = np.reshape(characters, (len(characters), 35*60))
    print(flatten_chars.dtype)
    print(labels.dtype)

    mean_training_accuracy=0
    mean_test_accuracy=0

    k = 15
    iter = 10
    kf = RepeatedKFold(n_splits=k, n_repeats=iter, random_state=random.randint(1,2652124))
    for i, (train, test) in enumerate(kf.split(flatten_chars)):
        train = np.array(train)
        test = np.array(test)
        X_train, X_test = flatten_chars[train], flatten_chars[test]
        y_train, y_test = labels[train], labels[test]

        print("\n\n------------------ K-fold test " + str(i) + " ------------------")
        print("Building & training SVM classifier...")
        svc = build_SVM(X_train, y_train);

        print("\t\t---------Validating training data---------")
        pred = svc.predict(X_train)
        training_accuracy = 0
        for idx, p in enumerate(pred):
            if p == y_train[idx]:
                training_accuracy += 1
        training_accuracy /= len(pred)
        training_accuracy *= 100
        mean_training_accuracy += training_accuracy
        print("Training Accuracy: " + str(training_accuracy) + "%")

        print("\n\t\t---------Predicting test data---------")
        pred = svc.predict(X_test)
        false_predictions = {}
        false_predictions_list = {}
        possible_labels = "0123456789ABCDEFGHJKLMNOPRSTUVWXYZ"
        for label in possible_labels:
            false_predictions[label] = 0
            false_predictions_list[label] = []

        test_accuracy = 0
        for idx, p in enumerate(pred):
            if p == y_test[idx]:
                test_accuracy += 1
            else:
                false_predictions[y_test[idx]] += 1
                false_predictions_list[y_test[idx]].append(p)
        test_accuracy /= len(pred)
        test_accuracy *= 100
        mean_test_accuracy += test_accuracy
        print("Testing Accuracy: " + str(test_accuracy) + "%")
        # print("Character Mispredictions:")
        # for label in false_predictions:
        #     if false_predictions[label] != 0:
        #         print(label + ": " + str(false_predictions[label]), str(false_predictions_list[label]))
    print("\n\n--------------K fold results---------------")
    print("Average training accuracy: " + str(mean_training_accuracy/(k*iter)) + "%")
    print("Average test accuracy: " + str(mean_test_accuracy/(k*iter)) + "%")

# Export pre-trained SVM classifer
def extract_svm_main():
    # Image data folder e.g. './images/'
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True)
    args = vars(ap.parse_args())

    print("Importing training data...")
    (X_train, y_train) = import_k_fold(args)
    X_train = np.reshape(X_train, (len(X_train), 35*60))

    print("Building & training SVM classifier...")
    svc = build_SVM(X_train, y_train);

    print("\t\t---------Validating training data---------")
    pred = svc.predict(X_train)
    training_accuracy = 0
    for idx, p in enumerate(pred):
        if p == y_train[idx]:
            training_accuracy += 1
    training_accuracy /= len(pred)
    training_accuracy *= 100
    print("Training Accuracy: " + str(training_accuracy) + "%")

    print("Exporting model to ANPR_SVM_vX.joblib")
    dump(svc, './trained_models/ANPR_SVM_vX.joblib')

extract_svm_main() # Start ANPR
