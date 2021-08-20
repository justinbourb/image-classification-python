from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import numpy as np
from sklearn.model_selection import train_test_split
from Python.train_test import import_feature_and_labels
from Python.helpers.tunable_parameters import test_size, seed, num_trees

def example():
	X, y = make_classification(n_samples=1000, n_features=4,
	                           n_informative=2, n_redundant=0,
	                           random_state=0, shuffle=False)
	clf = RandomForestClassifier(max_depth=2, random_state=0)
	clf.fit(X, y)

	print(X)
	# throws error with single brackets
	print(clf.predict([[-1, -1, -1, -1]]))


def something():
	global_features, global_labels = import_feature_and_labels()

	# split the training and testing data
	# this function is provided by the scikit-learn library
	# it uses k-fold cross validation to split the data into 9 training sets and 1 test set
	(trainDataGlobal, testDataGlobal, trainLabelsGlobal, testLabelsGlobal) = train_test_split(np.array(global_features),
	                                                                                          np.array(global_labels),
	                                                                                          test_size=test_size,
	                                                                                          random_state=seed)
	# create the model - Random Forests
	clf = RandomForestClassifier(n_estimators=num_trees, random_state=seed)
	# fit the training data to the model
	clf.fit(trainDataGlobal, trainLabelsGlobal)

	a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
	print(np.where(a == 3))


	for item in testDataGlobal:
		# print(clf.predict([item]))
		pass
something()