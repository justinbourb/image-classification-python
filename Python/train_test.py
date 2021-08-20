# -----------------------------------
# TRAINING OUR MODEL
# -----------------------------------
import numpy as np
import os
import glob
# cv2 requires pip install opencv-python
import cv2
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
# from sklearn.externals import joblib # does not work

# import these functions from generate_global_descriptors.py
from Python.helpers.generate_global_descriptors import fd_hu_moments
from Python.helpers.generate_global_descriptors import fd_haralick
from Python.helpers.generate_global_descriptors import fd_histogram
# import this variable from generate_global_descriptors.py
from Python.helpers.generate_global_descriptors import fixed_size

# to visualize results
import matplotlib.pyplot as plt

# import shared function
from Python.helpers.import_data import import_feature_and_labels
# import shared tunable parameters
from Python.helpers.tunable_parameters import num_trees, test_size, seed, train_path, test_path

# this silences warnings such a depreciation warnings
warnings.filterwarnings('ignore')


def main_function():
    """
    Purpose: This is the main function to execute the script.  This file will perform image classification
        analysis.
    :return: nothing
    """
    # get the training labels
    train_labels = os.listdir(train_path)

    # sort the training labels
    train_labels.sort()

    if not os.path.exists(test_path):
        os.makedirs(test_path)


    # import the feature vector and trained labels
    global_features, global_labels = import_feature_and_labels()

    # verify the shape of the feature vector and labels
    print("[STATUS] features shape: {}".format(global_features.shape))
    print("[STATUS] labels shape: {}".format(global_labels.shape))

    print("[STATUS] training started...")

    # split the training and testing data
    # this function is provided by the scikit-learn library
    # it uses k-fold cross validation to split the data into 9 training sets and 1 test set
    (trainDataGlobal, testDataGlobal, trainLabelsGlobal, testLabelsGlobal) = train_test_split(np.array(global_features),
                                                                                              np.array(global_labels),
                                                                                              test_size=test_size,
                                                                                         random_state=seed)


    # -----------------------------------
    # TESTING OUR MODEL
    # -----------------------------------

    # create the model - Random Forests
    clf = RandomForestClassifier(n_estimators=num_trees, random_state=seed)

    # fit the training data to the model
    clf.fit(trainDataGlobal, trainLabelsGlobal)

    # loop through the test images
    for file in glob.glob(test_path + "/*.jpg"):
        # read the image
        image = cv2.imread(file)

        # resize the image
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

        # scale features in the range (0-1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        rescaled_feature = scaler.fit_transform(global_feature)

        # predict label of test image
        prediction = clf.predict(rescaled_feature.reshape(1, -1))[0]

        # show predicted label on image
        cv2.putText(image, train_labels[prediction], (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)

        # display the output image
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.show()


if __name__ == "__main__":
    main_function()

    '''
    TODO:
        1) move copy the test files into \\dataset\\test directory based on the segmentation calculated by this file
        2) assumptions: 
            i) segmented labels is similar to labels.csv that was created with the 
                generate_global_descriptors.py file format of #, cluster (0, 0).  
            ii) The images files are named 1-80.  The labels are named 0-79, 80-159, etc so they are off by 1 and
                a factor of 80.
            iii) This information can be used to copy the test files into the test directory.
            iv) I need some way to determine which # = which name 
            
    '''