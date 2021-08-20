# Image Classification using Python and Machine Learning
Originally written by [Gogul Ilango](https://github.com/Gogul09)  
8.19.21 Forked and Modified version by [J. Bourbonniere](https://github.com/justinbourb)  
This repo contains the code to perform a simple image classification task using Python and Machine Learning. We will apply global feature descriptors such as Color Histograms, Haralick Textures and Hu Moments to extract features from FLOWER17 dataset and use machine learning models to learn and predict.

[UPDATE]
Now, you can simply run `download_and_organize_flowers_files.py` script to download and organize training data for this project. Also, I have updated the code to support only Python 3+ as Python 2+ faces end of life.

## Summary of the project
* Global Feature Descriptors such as Color Histograms, Haralick Textures and Hu Moments are used on University of Oxford's FLOWER17 dataset.
* Classifiers used are Logistic Regression, Linear Discriminant Analysis, K-Nearest Neighbors, Decision Trees, Random Forests, Gaussian Naive Bayes and Support Vector Machine.

## Usage 

1. `python download_and_organize_flowers_files.py` - Downloads Flowers17 Dataset and organizes training set in disk.
2. `python generate_global_descriptors.py` - Extracts global features from training set and stores it in disk.
3. `compare_models.py` - (Not Required) Displays the performance of various models when applied to this problem.
4. `python train_test.py` - Predicts the image class using the trained model.

Tutorial for this project is available at - [Image Classification using Python and Machine Learning](https://gogul09.github.io/software/image-classification-python)
Github Project Source [Original Author's Repository](https://github.com/Gogul09/image-classification-python)

## 8.19.21 Forked changes 
This project has been forked from the above tutorial. Code cleanup and 
documentation has been undertaken by J. Bourbonniere.  This project has been completely
redesigned and rebuilt from the ground up.  Digging into and understanding other developers
code is a key skill.  If I can rebuild the code and it still functions, that proves
I understand the code.

I would like to extend a huge thank you to the tutorial author Mr. Gogul Ilango
for writing an excellent guide.  Any modification or commentary on his work is not ment
as criticism.

## Changes vs. source code
* refactored entire code base
    * organized code into methods
    * organized code into separate files
* extracted methods for each file (previous one giant file without any methods)
* Implemented DRY code base
    * extracted methods to their own files for reuse
    * created shared parameters across code base
* Created prediction model functionality
    * source code does not produce any predictions
        * train_test.py relies on a user manually placing test images into 
        the test directory
    * refactored version automatically tests the segmented test data
        created by train_test.py

## Lessons learned
* image classification in Python using sklearn, opencv, pandas, matplotlib and mahotas
* image classification techniques (global and local descriptors)
* working with, debugging and understanding others code
    * line by line analysis of code
* good documentation practices
* working with h5py data sets for big data
* experience with machine learning development tools