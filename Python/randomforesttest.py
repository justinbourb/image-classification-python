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

# need to compare test_prediction_feature which works correctly and train_test.py which does not
# why does the rescale_features_func fail on the test data?


# rescaled features
# image 1, image 2, image 3 - all global features are set to 0
# something is wrong with the rescaled_features
# [[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.,  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.,  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.,  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.,  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.,  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.,  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.,  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.,  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.,  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.,  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.,  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.,  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.,  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. ...

# global features
# image 1
# [1.01004122e-02 2.08000001e-02 1.13215916e-01 2.18870789e-01, 2.14277096e-02 1.21147878e-01 2.66947597e-01 1.12987660e-01, 1.02716056e-03 8.63385499e-02 3.93459536e-02 1.32104252e-02, 3.43813449e-02 1.26397805e-02 7.17015117e-02 8.87352601e-03, 8.27434880e-04 3.14995907e-02 7.61810737e-03 5.19286701e-03, 2.87034307e-02 6.79067243e-03 2.70770919e-02 1.10134436e-02, 3.42386833e-04 3.76625522e-03 0.00000000e+00 3.16707836e-03, 1.78897120e-02 9.98628326e-03 1.92021951e-02 8.70233215e-03, 1.14128947e-04 1.14128947e-04 0.00000000e+00 2.99588474e-03, 2.97876559e-02 4.01448570e-02 3.12427990e-02 1.06425239e-02, 0.00000000e+00 0.00000000e+00 0.00000000e+00 4.85048024e-03, 2.10853238e-02 3.89750339e-02 2.36532241e-02 8.64526816e-03, 0.00000000e+00 0.00000000e+00 0.00000000e+00 2.08285335e-03, 1.18123461e-02 1.99440327e-02 1.88312761e-03 0.00000000e+00, 0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00, 5.70644734e-05 0.00000000e+00 0.00000000e+00 0.00000000e+00, 1.74046645e-03 4.080109...

# image 2
# [3.50401364e-02 1.73305161e-04 4.44094476e-04 2.92452460e-04, 0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00, 1.48825804e-02 8.12367944e-04 9.20683669e-04 1.61390426e-03, 0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00, 9.15267877e-03 1.72222010e-03 7.83122703e-03 6.93220645e-04, 0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00, 4.60341852e-03 3.19531397e-03 7.39796413e-03 2.20964081e-03, 0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00, 4.41928161e-03 2.98951403e-03 3.92102916e-03 1.25646242e-03, 5.41578629e-05 0.00000000e+00 0.00000000e+00 0.00000000e+00, 3.17365071e-03 2.48043006e-03 1.91718829e-03 1.70055684e-03, 1.73305161e-03 3.79105040e-04 8.99020524e-04 6.70474349e-03, 2.16631452e-03 5.17749181e-03 3.11949290e-03 5.10167051e-03, 1.15031302e-02 8.79523717e-03 7.08384812e-02 6.76973313e-02, 5.90320723e-03 1.82295367e-02 1.83378533e-02 1.38427494e-02, 2.97326669e-02 4.00356591e-01 8.63687932e-01 2.86603402e-02, 5.63241774e-04 3.249471...

# image 3
# [1.84988484e-01 0.00000000e+00 0.00000000e+00 0.00000000e+00, 0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00, 1.96086999e-04 0.00000000e+00 7.84348013e-05 1.37260897e-04, 1.17652206e-04 2.74521793e-04 5.29434881e-04 6.31400151e-03, 9.41217644e-04 3.92173999e-04 1.49026117e-03 2.41187005e-03, 4.00017481e-03 1.14710899e-02 2.40598749e-02 1.67458300e-02, 2.70600058e-03 3.64721823e-03 7.15717580e-03 1.59222651e-02, 3.42171825e-02 7.26894513e-02 4.62373160e-02 3.33347917e-03, 1.03730028e-02 1.11181326e-02 2.40010489e-02 4.10606191e-02, 8.06701928e-02 8.48076269e-02 2.44716574e-02 2.43147882e-03, 1.94910485e-02 3.32759656e-02 5.98457530e-02 7.32581019e-02, 9.94553268e-02 7.52189755e-02 3.64329666e-02 1.88831780e-02, 2.95699202e-02 7.37875402e-02 1.06808588e-01 1.08592980e-01, 9.04745460e-02 1.06573284e-01 1.50202647e-01 1.39417857e-01, 3.25465202e-01 2.90600926e-01 2.29519844e-01 2.03852043e-01, 2.37579018e-01 3.01542580e-01 4.51137364e-01 4.01194006e-01, 0.00000000e+00 0.000000...