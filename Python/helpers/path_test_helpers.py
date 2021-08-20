import os
from Python.helpers.tunable_parameters import num_trees, test_size, seed, train_path, test_path

if __name__ == "__main__":
	print(test_path)
	print(os.path.isdir(test_path))