# Image Classification using Python and Machine Learning
Originally written by [Gogul Ilango](https://github.com/Gogul09)  
8.19.21 Forked and Modified version by [J. Bourbonniere](https://github.com/justinbourb)  
This repo contains the code to perform a simple image classification task using Python and Machine Learning. We will apply global feature descriptors such as Color Histograms, Haralick Textures and Hu Moments to extract features from FLOWER17 dataset and use machine learning models to learn and predict.

[UPDATE]
Now, you can simply run `organize_flowers_files.py` script to download and organize training data for this project. Also, I have updated the code to support only Python 3+ as Python 2+ faces end of life.

## Summary of the project
* Global Feature Descriptors such as Color Histograms, Haralick Textures and Hu Moments are used on University of Oxford's FLOWER17 dataset.
* Classifiers used are Logistic Regression, Linear Discriminant Analysis, K-Nearest Neighbors, Decision Trees, Random Forests, Gaussian Naive Bayes and Support Vector Machine.

## Usage 

1. `python organize_flowers_files.py` - Downloads Flowers17 Dataset and organizes training set in disk.
2. `python generate_global_descriptors.py` - Extracts global features from training set and stores it in disk.
3. `python train_test.py` - Predicts the image class using the trained model.

Tutorial for this project is available at - [Image Classification using Python and Machine Learning](https://gogul09.github.io/software/image-classification-python)
Github Project Source [Original Author's Repository](https://github.com/Gogul09/image-classification-python)

## 8.19.21 Forked changes 
This project has been forked from the above tutorial. Code cleanup and 
documentation has been undertaken by J. Bourbonniere.

## Lessons learned
* image classification in Python using sklearn, opencv, pandas, matplotlib and mahotas
* image classification techniques (global and local descriptors)
* working with and understanding others code
* good documentation practices