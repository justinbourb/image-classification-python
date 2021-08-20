# -----------------------------------
# GLOBAL FEATURE EXTRACTION
# -----------------------------------
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import mahotas
import cv2
import os
import h5py
import pandas as pd

# import tunable-parameters
from Python.helpers.tunable_parameters import fixed_size, train_path, images_per_class, bins, h5_labels_path, h5_data_path


# feature-descriptor-1: Hu Moments
def fd_hu_moments(image):
    """
    Purpose: This function extracts the Hu Moments feature of a flattened image. Hu Moments are used to
    describe the shape of an image.
    :param image: a jpg image file
    :return: Hu Moments feature
    Requirements:
        1) cv2 (opencv-python) expects images to be in grayscale
        2) cv2 (opencv-python) expects images to flatted before HuMoments analysis
            a) we are using numpy.flatten()
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature


# feature-descriptor-2: Haralick Texture
def fd_haralick(image):
    """
    Purpose: This function provides Haralick Texture analysis for images.
    :param image: a jpg image
    :return: The haralick analysis
    Requirements:
        1) The haralick() function from the mahotas library expects images to be in grayscale.
    """
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # compute the haralick texture feature vector
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    # return the result
    return haralick


# feature-descriptor-3: Color Histogram
def fd_histogram(image, mask=None):
    """
    Purpose: This function will provide histogram color analysis on an image.
    :param image: a jpg image
    :param mask: an OpenCV calcHist requirement
    :return: a flattened histogram
    Requirements / Notes:
        1) cv2 (opencv-python), numpy flatten()
        2) cv2.calcHist arguments image, channels, mask, histSize (bins) and ranges for each channel
            (typically 0-256)
            a) A mask is a border to place over a image when we want to only analyze the portion of the
                image that fits inside the border
            b) bins = 8, I'm not sure the significance of this value
            c) channels 0,1,2 = R,G,B ?? The opencv docs do not clarify this parameter for me.
    """
    # convert the image to HSV color-space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # compute the color histogram
    hist = cv2.calcHist([image], [0, 1, 2], mask, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    # normalize the histogram
    cv2.normalize(hist, hist)
    # return the histogram
    return hist.flatten()


def save_output_to_csv(data, file_name):
    """
    Purpose: This functions converts the data to a pandas dataframe and saves the data as a csv file.
    I've enabled saving the data as a csv file to satisfy my curiosity regarding which data is being stored
    as a h5 file.  CSV files are human readable, thus I will save a copy of the data as CSV.

    This function is not required for the analysis.
    :param data: the data to be stored
    :param file_name: a string, the file name
    :return: nothing
    Output: creates a csv file
    Results:
        1) The labels file is equivalent to cluster assignment.  There are 17 flowers being analyzed for this
            example project.  Thus each image trained is given a label from 0-16 which relates to each flower
            type that was pre-classified for training.
        2) The data file is a concatenation of the color histogram, hu moments and haralick texture global
            features analysis.  The data is normalized to fit the range 0-1 to help improve model accuracy.
            Data that is not normalized can skew the results by containing large values.  Each row contains
            532 columns of data.  There are 1360 rows of data.
    """
    numpy_data = np.array(data)
    pandas_data = pd.DataFrame(numpy_data)
    path = "output/" + file_name
    pandas_data.to_csv(path)


def generate_descriptors(train_labels):
    """
    Purpose: This function generates the global descriptors for the images provided.  It combines three methods
    Hu Moments, Haralick Texture and Color Histograms into one global descriptor per image.
    :param train_labels: list of folder names to search for images
    :return global_features, labels: a numpy array of global features, a numpy array of labels
    """
    # empty lists to hold feature vectors and labels
    global_features = []
    labels = []
    # loop over the training data sub-folders
    for training_name in train_labels:
        # join the training data path and each species training folder
        joined_dir = os.path.join(train_path, training_name)

        # get the current training label
        current_label = training_name

        # loop over the images in each sub-folder
        for x in range(1, images_per_class + 1):
            # get the image file name
            file = joined_dir + "/" + str(x) + ".jpg"

            # read the image and resize it to a fixed-size
            image = cv2.imread(file)
            image = cv2.resize(image, fixed_size)

            ####################################
            # Global Feature extraction
            ####################################
            fv_hu_moments = fd_hu_moments(image)
            fv_haralick = fd_haralick(image)
            fv_histogram = fd_histogram(image)

            ###################################
            # Concatenate global features
            ###################################
            global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])

            # update the list of labels and feature vectors
            labels.append(current_label)
            global_features.append(global_feature)

        print("[STATUS] processed folder: {}".format(current_label))
    return global_features, labels


def save_h5_and_csv(rescaled_features, target):
    """
    Purpose: This function saves the data as a h5 file (used for analysis) and as a csv file
        (used for human readable format).
    :param rescaled_features:
    :param target:
    :return: nothing
    """
    # save the feature vector using HDF5
    h5f_data = h5py.File(h5_data_path, 'w')
    h5f_data.create_dataset('dataset_1', data=np.array(rescaled_features))
    # save the labels using HDF5
    h5f_label = h5py.File(h5_labels_path, 'w')
    h5f_label.create_dataset('dataset_1', data=np.array(target))
    # close the HDF5 files
    h5f_data.close()
    h5f_label.close()
    # save the data as csv for a human readable format (not required for analysis)
    save_output_to_csv(target, 'labels.csv')
    save_output_to_csv(rescaled_features, 'data.csv')


def encode_and_scale(global_features, labels):
    """
    Purpose: This function will encode the labels (change the labels from a string name to number 0-16).
        This function will scale/normalize the features in a range from 0-1.  This is required for accurate
        machine learning.  Large values will skew the results.
    :param global_features:
    :param labels:
    :return rescaled_features, encoded_labels: the scaled features, the encoded targets
    """
    # encode the target labels
    # This uses from sklearn.preprocessing import LabelEncoder
    # targetNames = np.unique(labels) # this line is not used, it would provide only unique names
    le = LabelEncoder()
    encoded_labels = le.fit_transform(labels)
    print("[STATUS] training labels encoded...")
    # scale features in the range (0-1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    rescaled_features = scaler.fit_transform(global_features)
    print("[STATUS] feature vector normalized...")
    print("[STATUS] target labels: {}".format(encoded_labels))
    print("[STATUS] target labels shape: {}".format(encoded_labels.shape))
    return rescaled_features, encoded_labels


def main_function():
    """
    Purpose: This file will analyze the global descriptors for images undergoing image classification.
        Image classification involves global descriptors (features of the entire image) and local descriptors
        (features of an area of interest within the image).  This file uses color histograms to analyze color,
        Hu Moments to analyze shape and Haralick Textures to analyze texture.

        The resulting analysis is output to a HDF5 data.h5 file and labels.h5 file.  The h5 file format was
        chosen because it is designed for large datasets.
    :return: nothing
    Output:  HDF5 Data and labels from all images provided for processing.
    """
    # get the training labels
    train_labels = os.listdir(train_path)

    # sort the training labels
    train_labels.sort()
    # Expected output:
    # ['bluebell', 'buttercup', 'coltsfoot', 'cowslip', 'crocus', 'daffodil', 'daisy', 'dandelion', 'fritillary', 'iris', 'lilyvalley', 'pansy', 'snowdrop', 'sunflower', 'tigerlily', 'tulip', 'windflower']

    global_features, labels = generate_descriptors(train_labels)

    print("[STATUS] completed Global Feature Extraction...")

    # get the overall feature vector size
    print("[STATUS] feature vector size {}".format(np.array(global_features).shape))

    # get the overall training label size
    print("[STATUS] training Labels {}".format(np.array(labels).shape))

    # encode and scale features and labels
    rescaled_features, encoded_labels = encode_and_scale(global_features, labels)

    save_h5_and_csv(rescaled_features, encoded_labels)

    print("[STATUS] end of training..")


# -------------------------
# MAIN FUNCTION
# -------------------------
if __name__ == '__main__':
    main_function()
