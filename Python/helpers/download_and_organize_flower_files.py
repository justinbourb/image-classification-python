# -----------------------------------------
# DOWNLOAD AND ORGANIZE FLOWERS17 DATASET
# -----------------------------------------
import os
import glob
import tarfile
import urllib.request


def download_dataset(filename, url, work_dir):
	"""
	Purpose: This function will download the data from the specified location.
	:param filename: File to download
	:param url: Url to download from
	:param work_dir: working directory to download to
	:return: nothing
	"""
	if not os.path.exists(filename):
		print("[INFO] Downloading flowers17 dataset....")
		filename, _ = urllib.request.urlretrieve(url + filename, filename)
		statinfo = os.stat(filename)
		print("[INFO] Successfully downloaded " + filename + " " + str(statinfo.st_size) + " bytes.")
		extract_tar_file(filename, work_dir)


def jpg_files(members):
	"""
	Purpose: This function finds the jpg files inside the compressed tar file and returns them using yield.
	:param members: a tar file
	:return: yield tarinfo
	Notes: yield is similar to return in Python, except it allows returning
	multiple values.  Each value is processed before proceeding to the next.
	"""
	for tarinfo in members:
		if os.path.splitext(tarinfo.name)[1] == ".jpg":
			yield tarinfo


def extract_tar_file(fname, path):
	"""
	Purpose: This function extracts a tar file
	:param fname: file name
	:param path: path
	:return: nothing
	Outputs: the extracted file
	"""
	tar = tarfile.open(fname)
	tar.extractall(path=path, members=jpg_files(tar))
	tar.close()
	print("[INFO] Dataset extracted successfully.")


def main_function():
	"""
	Purpose: This function is the main body of the script.  It's purpose is to download,
	extract the tar files and organize the images in the working directory for further analysis.
	Organization is for human ease only.
	:return: nothing
	Output: Downloads, extracts and organizes the specified tar file by creating subdirectories.
	Requirements:
		1) This script assumes the source data is still being hosted at the url.
		2) This is designed to run on Windows 10
	Notes: I was unable to make Pycharm resolve mixed tab and space indentation pep formatting violations.
	"""
	# setup where we will get our data
	# The extracted size is 118 MB
	flowers17_url = "http://www.robots.ox.ac.uk/~vgg/data/flowers/17/"
	flowers17_name = "17flowers.tgz"
	train_dir = "../../dataset"

	# create the directory if it doesn't exist
	if not os.path.exists(train_dir):
		os.makedirs(train_dir)

	# download the data
	download_dataset(flowers17_name, flowers17_url, train_dir)
	if os.path.exists(train_dir + "\\jpg"):
		os.rename(train_dir + "\\jpg", train_dir + "\\train")

	# get the class label limit
	class_limit = 17

	# take all the images from the dataset
	image_paths = glob.glob(train_dir + "\\train\\*.jpg")

	# variables to keep track
	label = 0
	i = 0
	j = 80

	# flower17 class names
	class_names = ["daffodil", "snowdrop", "lilyvalley", "bluebell", "crocus",
	               "iris", "tigerlily", "tulip", "fritillary", "sunflower",
	               "daisy", "coltsfoot", "dandelion", "cowslip", "buttercup",
	               "windflower", "pansy"]

	# loop over the class labels
	for x in range(1, class_limit + 1):
		# create a folder for that class
		os.makedirs(train_dir + "\\train\\" + class_names[label])

		# get the current path
		cur_path = train_dir + "\\train\\" + class_names[label] + "\\"

		# loop over the images in the dataset
		for index, image_path in enumerate(image_paths[i:j], start=1):
			original_path = image_path
			# image_path = image_path.split("\\") # This line is not used
			image_file_name = str(index) + ".jpg"
			os.rename(original_path, cur_path + image_file_name)

		i += 80
		j += 80
		label += 1


# -------------------------
# MAIN FUNCTION
# -------------------------
if __name__ == '__main__':
	main_function()
