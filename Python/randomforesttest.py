import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import numpy as np
from sklearn.model_selection import train_test_split
from Python.helpers.import_data import import_feature_and_labels
from Python.helpers.tunable_parameters import test_size, seed, num_trees, train_path
from Python.train_test import train_model

train_labels = os.listdir(train_path)
print(train_labels)
train_labels.sort()
print(train_labels)

def test_prediction_feature():
	"""
	Purpose: This function will verify the clf.predict() feature is working correctly on our data set.
		Each item's prediction is printed to the screen and the user can verify that different items are
		getting different predictions.
	:return: Nothing
	Output: Prints a list of predictions.
	"""
	global_features, global_labels = import_feature_and_labels()

	# split the training and testing data
	# this function is provided by the scikit-learn library
	# it uses k-fold cross validation to split the data into 9 training sets and 1 test set
	(trainDataGlobal, testDataGlobal, trainLabelsGlobal, testLabelsGlobal) = train_test_split(np.array(global_features),
	                                                                                          np.array(global_labels),
	                                                                                          test_size=test_size,
	                                                                                          random_state=seed)
	# create the model - Random Forests
	clf = train_model(trainDataGlobal, trainLabelsGlobal)

	# visually verify different predictions are being made
	for item in testDataGlobal:
		print(clf.predict([item]))

test_prediction_feature()