from Python.helpers.import_data import import_feature_and_labels
from sklearn.preprocessing import LabelEncoder
import numpy as np
features, labels = import_feature_and_labels()


encoded_to_name_dic = {0: 'bluebell', 1: 'buttercup', 2: 'coltsfoot', 3: 'cowslip', 4: 'crocus', 5: 'daffodil',
                       6: 'daisy', 7: 'dandelion', 8: 'fritillary', 9: 'iris', 10: 'lilyvalley', 11: 'pansy',
                       12: 'snowdrop', 13: 'sunflower', 14: 'tigerlily', 15: 'tulip', 16: 'windflower'}
le = LabelEncoder()
encoded_labels = le.fit_transform(labels)
names = ['bluebell', 'buttercup', 'coltsfoot', 'cowslip', 'crocus', 'daffodil', 'daisy', 'dandelion', 'fritillary', 'iris', 'lilyvalley', 'pansy', 'snowdrop', 'sunflower', 'tigerlily', 'tulip', 'windflower']
some_dict = {}
# for each unique value in encoded labels (expected 0-16)
# match the numbers to their original name and store the results in a dictionary
# example: {0:'bluebell', etc}
for label in np.unique(encoded_labels):
	some_dict[label] = names.pop(0)
print(some_dict)