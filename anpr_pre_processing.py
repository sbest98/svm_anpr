# Import necessary packages
import cv2
import os
import re
import sys
import imutils
import argparse
import numpy as np
from matplotlib import pyplot as plt
from natsort import natsorted, ns
from skimage.segmentation import clear_border

# Pre-processing mode types
GENERATE_DATASET = 0
FIND_SVM_CHARACTERS = 1

"""
LocateAndDivideNumberPlate: Class that locates and sub divides the
                            characters from the vehicle number plate and
                            converts them into sub-images for SVM
"""
class LocateAndDivideNumberPlate:
    def __init__(self, path_to_images = './images/', debug = False):
        self.path_to_images = path_to_images
        self.global_char_count = 0
        self.char_count = 0
        self.debug = debug

    def show_image(self, title, image, waitKey=False):
        if self.debug:
            cv2.imshow(title, image)
            if waitKey:
                cv2.waitKey(0)

    def find_and_divide(self, image_path, pp_mode, border_test = False):
        image = cv2.imread(self.path_to_images + image_path)
        if image is not None:
            image = imutils.resize(image, width=600)
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            contour_list_v1 = self.locate_number_plate(image_gray, 12)
            contour_list_v2 = self.locate_number_plate_v2(image_gray, 12)

            if not border_test:
                characters_segmented, character_list, character_positions = self.locate_number_plate_characters(contour_list_v1
                                                                                                                + contour_list_v2,
                                                                                                                False,
                                                                                                                pp_mode)

                if not characters_segmented:
                    # Try clearing border pixels on contour
                    characters_segmented, character_list, character_positions = self.locate_number_plate_characters(contour_list_v1
                                                                                                + contour_list_v2,
                                                                                                True,
                                                                                                pp_mode)
            elif border_test:
                # Fall back test -  clearing border pixels on contour
                characters_segmented, character_list, character_positions = self.locate_number_plate_characters(contour_list_v1
                                                                                            + contour_list_v2,
                                                                                            True,
                                                                                            pp_mode)

            if not characters_segmented:
                # Failure!
                print("Unable to locate or segment number plate characters!")

            if pp_mode == FIND_SVM_CHARACTERS:
                return characters_segmented, character_list, character_positions
        else:
            sys.exit("Image does not exist in ./images/test_images/ directory")

    def locate_number_plate_v2(self, image, num_contours=5):
        cv2.destroyAllWindows()
        bil = cv2.bilateralFilter(image, 15, 50, 75);
        self.show_image("Bilateral Filter", bil)

        clahe_filter = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
        clahe = clahe_filter.apply(bil)
        self.show_image("Adaptive HE", clahe)

        disc_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
        square_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,4))
        opening = cv2.morphologyEx(clahe, cv2.MORPH_OPEN, disc_kernel)
        self.show_image("Circular Opening", opening)

        ahe_minus_open = clahe - opening
        self.show_image("AHE - Opening", ahe_minus_open)

        binarise = cv2.threshold(ahe_minus_open, 0, 255,
            cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        self.show_image("Binarise", binarise)

        sobel_dy = cv2.Sobel(binarise, ddepth=cv2.CV_32F,
            dx=0, dy=1, ksize=-1)
        sobel_dx = cv2.Sobel(binarise, ddepth=cv2.CV_32F,
            dx=0, dy=1, ksize=-1)
        abs_sobel_x = cv2.convertScaleAbs(sobel_dx)
        abs_sobel_y = cv2.convertScaleAbs(sobel_dy)
        sobel = cv2.addWeighted(abs_sobel_x, 0.5, abs_sobel_y, 0.5, 0)
        self.show_image("Sobel", sobel)

        dilate = cv2.dilate(sobel, None, iterations=1)
        self.show_image("Dilate", dilate)

        dilate_copy = dilate.copy()
        h, w = dilate_copy.shape[:2]
        flood_mask = np.zeros((h+2, w+2), np.uint8)
        cv2.floodFill(dilate_copy, flood_mask, (0,0), 255);
        floodfill_bitwise = cv2.bitwise_not(dilate_copy)
        floodfill_im = dilate | floodfill_bitwise
        self.show_image("Fill plate", floodfill_im)

        erode = cv2.erode(floodfill_im, None, iterations=1)
        self.show_image("Erode", erode, True)

        # Find the contours in the image
        contours = cv2.findContours(erode.copy(), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:num_contours]
        contour_list = []
        for c in contours:
            (x, y, w, h) = cv2.boundingRect(c)
            ratio = float(w/h)
            if ratio >= 2.9 and ratio <= 6.2:
                contour_image = image[y:y + h, x:x + w]
                contour_list.append(contour_image)
                self.show_image("Plate Contour", contour_image)
                cv2.waitKey(0)

        return contour_list

    def locate_number_plate(self, image, num_contours=5):
        cv2.destroyAllWindows()
        # Blackhat morphological operation
        #   - reveal dark regions (i.e., text) on light backgrounds
        rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, (30 , 5))
        blackhat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, rectKern)
        self.show_image("Blackhat", blackhat)

        # Closing then thresholding
        #   - find light regions
        squareKern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        light = cv2.morphologyEx(image, cv2.MORPH_CLOSE, squareKern)
        light = cv2.threshold(light, 0, 255,
            cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        #light = cv2.erode(light, None, iterations=1)
        self.show_image("Light Regions", light)

        # compute the Scharr gradient of blackhat
        #   - in x-direction
        #   - scale back to [0,255]
        gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F,
            dx=1, dy=0, ksize=-1)
        gradX = np.absolute(gradX)
        (minVal, maxVal) = (np.min(gradX), np.max(gradX))
        gradX = 255 * ((gradX - minVal) / (maxVal - minVal))
        gradX = gradX.astype("uint8")
        self.show_image("Scharr", gradX)

        # 1. blur the gradient representation
        # 2. applying a closing
        # 2. binarise
        gradX = cv2.GaussianBlur(gradX, (5, 5), 0)
        gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKern)
        thresh = cv2.threshold(gradX, 0, 255,
            cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        self.show_image("Grad Thresh", thresh)

        # Erode and dilate
        #   - remove noise
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)
        self.show_image("Grad Erode/Dilate", thresh)

        # take the bitwise AND between the threshold result and the
        # light regions of the image
        thresh = cv2.bitwise_and(thresh, thresh, mask=light)
        thresh = cv2.dilate(thresh, None, iterations=2)
        thresh = cv2.erode(thresh, None, iterations=1)
        self.show_image("Final", thresh, True)

        # Find the contours in the image
        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:num_contours]
        contour_list = []
        for c in contours:
            (x, y, w, h) = cv2.boundingRect(c)
            ratio = float(w/h)
            if ratio >= 2.9 and ratio <= 6.2:
                contour_image = image[y:y + h, x:x + w]
                contour_list.append(contour_image)
                self.show_image("Plate Contour", contour_image)
                cv2.waitKey(0)
        return contour_list

    def locate_number_plate_characters(self, contour_list, clear_border_pixels=False, pp_mode=0, num_contours=15):
        self.char_count = 0
        print("Locating characters...")
        cv2.destroyAllWindows()
        for image in contour_list:
            # Reveal dark regions on light regions (text on plate)
            rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 3))
            blackhat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, rectKern)
            self.show_image("Blackhat", blackhat)

            # Binarise image
            binarised = cv2.threshold(blackhat, 0, 255,
                cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            self.show_image("Binarised Image", binarised)

            if clear_border_pixels:
                cb = clear_border(binarised)
                self.show_image("Clear border pixels", cb, True)

            # Find the contours in the image
            if clear_border_pixels:
                contours = cv2.findContours(cb.copy(), cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)
            else:
                contours = cv2.findContours(binarised.copy(), cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)
            contours = imutils.grab_contours(contours)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:num_contours]

            char_images = []
            char_positions = [] # [x,y,w,h]
            for c in contours:
                (x, y, w, h) = cv2.boundingRect(c)
                area = w*h
                ratio = float(h/w)
                contour_image = image[y:y + h, x:x + w]
                if ((ratio >= 1.2 and ratio <= 2.3) or (ratio >= 3.7 and ratio <= 6.0)) and (area > 30):
                    self.char_count += 1 # Character found
                    self.global_char_count += 1 # Character found
                    contour_image = cv2.threshold(contour_image, 0, 255,
                                            cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
                    char_images.append(cv2.resize(contour_image, (35,60)))
                    char_positions.append([x, y, w, h])
                    self.show_image("Plate Character", contour_image, True)

            if self.char_count < 7:
                if pp_mode == GENERATE_DATASET:
                    print(str(self.char_count) + " characters found")
                self.global_char_count -= self.char_count
                self.char_count = 0
            else:
                if pp_mode == GENERATE_DATASET:
                    print("Found all characters!")
                    for idx, image in enumerate(char_images):
                        cv2.imwrite(self.path_to_images + 'located_characters/char_' + str(self.global_char_count - self.char_count + idx) + '.jpg', image)
                    return True, [], []
                elif pp_mode == FIND_SVM_CHARACTERS:
                    return True, char_images, char_positions

        return False, [], [] # Unable to segment 7 characters from contour list

def extract_character_data_set():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", required=True)
    ap.add_argument("-d", required=True)
    arg = vars(ap.parse_args())

    pre_proc = LocateAndDivideNumberPlate(debug = int(arg["d"]))
    # Iterate through images and generate dataset
    car_images = natsorted(os.listdir('./images/' + arg["i"]))
    remove_folders = re.compile('^.*\.(jpg|png)$')
    car_images = [i for i in car_images if remove_folders.match(i)]
    print(car_images)
    for image in car_images:
        print("\nLocating and dividing number plate in: " + image)
        pre_proc.find_and_divide(arg["i"] + image, GENERATE_DATASET)

def extract_svm_characters(image_name, debug, border_test=False):
    pre_proc = LocateAndDivideNumberPlate(path_to_images = './images/test_images/', debug=debug)
    plate_found, image_list, position_list = pre_proc.find_and_divide(image_name,
                                                                      FIND_SVM_CHARACTERS,
                                                                      border_test)
    return plate_found, image_list, position_list
