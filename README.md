# Road segmentation from satellite imagery using U-net architecture

*TO CHANGE: all instances of ' ' that adresses a certain machine learning model, to the model we have actually used as our classifier*

## About the project

The aim of this project is to train a classifier, using Convolutional Neural Networks (CNN's), and in particular U-nets; that extract roads in a set of aerial images aquired from GoogleMaps. From each image provided to train the model, there is a ground-truth form of that image where each pixel is labeled as road or background.  We are also provided by a set of images, from where we will test our model by producing our classifier's ground-truth version of those images. Essentially the goal reduces to train our classifier to assign a label `road = 1`, `background = 0` to each pixel. 

To do so, we have done the following:

* Data augmentation to avoid over-fitting and to improve the robustness of the data and better the training of our U-net model. In      particular, performing random image manipulation with a certain probability `p` by making use of the python library `Augmentor`. Some of the transformations made are the subsequent:
    * Left or right rotations of maximum 20 degrees `p = 1`
    * Mirroring images left to right or top to bottom `p = 0.5`
    * Zooming with a 0.8 rate `p = 0.5`
    * Left or right shears of maximum 5 degrees `p = 0.3`
    * Distortions that can make roads appear 'wavy' `p = 0.5`

* Implementing a U-net classifier, a particular type of CNN and one of the most widely used models for image segmentation, and training it using the images we were given.
* Analyzing the model and using various performance metrics like training and validation loss, dice loss, binary focal loss, combination loss, intersection over union (IOU) score, and F1 score to keep improving our classifier. While comparing the results of our model with the real ground truth form of those images, which the model is trained on.
* Predicting perceivable drivable regions from images of GoogleMaps.

## Figure representation


## Project structure

The following represents our project structure:

- **[`addittional_data/`](./addittional_data.ipynb)**: Image manipulation to generate an extensive training data set

- **[`model/`](./model.ipynb)**: 'Machine learning model' 'classifier'

- **[`mask_to_submission/`](./mask_to_submission.py)**: Helper function converting from a visualization format to a submission format 

- **[`submission_to_mask/`](./submission_to_mask.py)**: Helper function converting from a submission format to a visualization format 

- **[`Project_Report/`](./CS433_Project2_SFS.pdf)**: Project Report

- **[`README`](./README.md)**: This README file


## Installation Instructions

In contrast to Project1, for Project2 we have had the need to install additional machine learning libraries apart from the basic ones. Having been familiriazed with the basic machine learning implementations we no longer had to create them, but simply use them from the more advanced libraries that provide them.

1. Python~=3.13.0
    
    You can either download [Python](https://www.python.org/downloads/) directly or through an [Anaconda](https://www.anaconda.com/download/) distribution.

The following libraries are downloaded assuming we have installed python through Anaconda.

2. numpy~=2.1.2
    
    Run on a python terminal `pip install numpy` and it will download the latest version.

3. matplotlib~=3.9.2
    
    Run on a python terminal `pip install matplotlib` and it will download the latest version.

4. sklearn~=1.5.2

    Run on a python terminal `pip install scikit-learn` and it will download the latest version.
    
5. pytorch~=2.5.1

    [Tutorial](https://www.youtube.com/watch?v=STYdcBIT9H8) on how to install PyTorch in Anaconda on both CPU and GPU, downloading CUDA for the latter.

6. tensorflow~=2.16.1

    [Tutorial](https://www.youtube.com/watch?v=QJjHc2iSeBc) on how to install TensorFlow in Anaconda.



## Known Issues

'To complete'
