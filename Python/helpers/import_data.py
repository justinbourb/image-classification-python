import os

import h5py
import numpy as np

from Python.helpers.tunable_parameters import h5_data_path, h5_labels_path


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