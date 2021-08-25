# -----------------------------------
# TRAINING OUR MODEL
# -----------------------------------
import h5py
import numpy as np
import os
import glob
# cv2 requires pip install opencv-python
import cv2
import warnings
from matplotlib import pyplot
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
# from sklearn.externals import joblib # does not work

# import these functions from generate_global_descriptors.py
from Python.helpers.generate_global_descriptors import fd_hu_moments
from Python.helpers.generate_global_descriptors import fd_haralick
from Python.helpers.generate_global_descriptors import fd_histogram
# import this variable from generate_global_descriptors.py
from Python.helpers.generate_global_descriptors import fixed_size

# to visualize results
import matplotlib.pyplot as plt

from Python.helpers.tunable_parameters import h5_data_path, h5_labels_path

# this silences warnings such a depreciation warnings
warnings.filterwarnings('ignore')

# --------------------
# tunable-parameters
# --------------------
num_trees = 100
test_size = 0.10
seed = 9
train_path = "../dataset/train"
test_path = "../dataset/test"
h5_data = 'output/data.h5'
h5_labels = 'output/labels.h5'
scoring = "accuracy"


def import_feature_and_labels():
    """
    Purpose: This function will open the h5 data and labels files and convert them into numpy arrays.
        The numpy arrays are returned.
    :return: a numpy array of the global features and global labels
    """
    # import the feature vector and trained labels
    h5f_data = h5py.File(h5_data_path, 'r')
    h5f_label = h5py.File(h5_labels_path, 'r')

    global_features_string = h5f_data['dataset_1']
    global_labels_string = h5f_label['dataset_1']

    global_features = np.array(global_features_string)
    global_labels = np.array(global_labels_string)

    h5f_data.close()
    h5f_label.close()
    return global_features, global_labels


def create_models():
    """
    Purpose: This function creates a list of the machine learning models to use and their associated
        function calls using the scilearn library.
    :return:a list of machine learning models to use
    """
    # create all the machine learning models
    models = []
    models.append(('LR', LogisticRegression(random_state=seed)))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier(random_state=seed)))
    models.append(('RF', RandomForestClassifier(n_estimators=num_trees, random_state=seed)))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC(random_state=seed)))
    return models


def run_and_plot_models(models, train_data_global, train_labels_global):
    """
    Purpose: This function runs 10-fold cross validation and plots the results.
    :param models: a dictionary of models and their methods
    :param train_data_global: training data
    :param train_labels_global: training labels
    :return: nothing
    Output: Plots the results
    Results:
        The accuracies are not so good. Random Forests (RF) gives the maximum accuracy of 64.38%.
        This is mainly due to the number of images we use per class. We need large amounts of data to get
        better accuracy. For example, for a single class, we at least need around 500-1000 images which is
        indeed a time-consuming task.  Due to computational constraints a smaller number of images has been
        analyzed.
    """
    # variables to hold the results and names
    results = []
    names = []

    # 10-fold cross validation
    # This runs all of the different models stored in the model variable
    for name, model in models:
        kfold = KFold(n_splits=10, random_state=seed)
        cv_results = cross_val_score(model, train_data_global, train_labels_global, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)

    # boxplot algorithm comparison
    fig = pyplot.figure()
    fig.suptitle('Machine Learning algorithm comparison')
    ax = fig.add_subplot(111)

    # create a boxplot of the test results
    pyplot.boxplot(results)

    # set the x ticks to be the model names
    ax.set_xticklabels(names)

    # set the x and y labels
    pyplot.ylabel("Accuracy Percentage")
    pyplot.xlabel("Analysis Performed")

    # display the plot
    # I am using PyCharm and have toggled the setting to show plots as a new window.
    # Go to Settings => search for Python Scientific. Uncheck the (only) box Show plots in tool windows.
    pyplot.show()


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

    # create all the machine learning models
    models = create_models()

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

    print("[STATUS] train and test data has been split...")
    print("Train data  : {}".format(trainDataGlobal.shape))
    print("Test data   : {}".format(testDataGlobal.shape))
    print("Train labels: {}".format(trainLabelsGlobal.shape))
    print("Test labels : {}".format(testLabelsGlobal.shape))

    # This function runs validation and plots the results
    run_and_plot_models(models, trainDataGlobal, trainLabelsGlobal)

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
