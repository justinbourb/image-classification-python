import os
from Python.helpers.tunable_parameters import h5_data_path, test_path, train_path, dirname

if __name__ == "__main__":
	example = (dirname.split('image-classification-python')[0]+train_path)
	print(test_path)
	print(os.path.isdir(test_path))




	# maybe I want string manipulation to join the paths where folder names match?