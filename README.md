# ES410 ITS: Automatic Number Plate Recognition (ANPR) tool

This is the ANPR component of the University of Warwick, School of Engineering, ES410 Intelligent Transport Systems project of 2020-2021.

### Prerequisites

This tool is built using Python 3. Please install Python before any of required the packages:
* OpenCV
* Natsorted
* Sklearn
* Skimage
* Imutils
* Matplotlib
* Joblib
  ```sh
  pip3 install <package-name>
  ```

## Directory Structure

```sh
.
├── images
│   ├── k_characters
│   │   ├── 0
│   │   ├── ...
│   │   └── Z
│   ├── test_images
└── trained_models
```

## Usage
### ANPR Tool
To run the ANPR tool, use the following command:
```sh
python3 anpr.py -d 0 -i test_image.png
```
Command options include:
```
-d 0 (debug disabled) or 1 (debug enabled)
-i (test image in ./images/test_images/)
```

### Training or Validating SVM Model
To k-fold validate the SVM model, run the following command:
```sh
python3 anpr_svm.py -m 1 -i ./images/
```
To train the SVM model, run the following command:
```sh
python3 anpr_svm.py -m 0 -i ./images/
```
Command options include:
```
-m 0 (train) or 1 (validate)
-i (image data folder where k_characters/ is)
```

The training data is not included in this repository. Please contact sbest98 for access.
