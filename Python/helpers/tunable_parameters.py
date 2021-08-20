"""
Purpose: This file contains the tunable parameters which are shared between project files.
Reasoning:
	1) Storing shared parameters in one place is DRY code. No repetition.
	2) If modification is desired, it can be one all in one place.  Digging through every file is not required.
"""

# --------------------
# tunable-parameters
# --------------------
import os

# using dirname and os.path.join relative path creates an absolute path
# this allows the path to be used from different file locations
# I am splitting the path on the root folder for this project, this assumes the root folder name will not change
# or if it does, it can be modified here in one place
dirname = os.path.dirname(__file__).split('image-classification-python')[0]

num_trees = 100
test_size = 0.10
seed = 9
train_path = os.path.join(dirname, "image-classification-python/dataset/train")
test_path = os.path.join(dirname, "image-classification-python/dataset/test")
h5_data_path = os.path.join(dirname, "image-classification-python/output/data.h5")
h5_labels_path = os.path.join(dirname, "image-classification-python/output/labels.h5")
scoring = "accuracy"
fixed_size = tuple((500, 500))
images_per_class = 80
bins = 8
