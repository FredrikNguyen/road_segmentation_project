# Road segmentation from satellite images using 'machine learning techniques'

*TO CHANGE: all instances of '' that adress machine learnign model, to the model we have actually used as our classifier*

## About the project

The aim of this project is to train a 'model classifier' that extracts roads in a set of aerial images aquired from GoogleMaps. From each image provided to train the model, there is a ground-truth form of that image where each pixel is labeled as road or background.  We are also provided by a set of images, from where we will test our model by producing our classifier's ground-truth version of those images. Essentially the goal reduces to train a classifier to assign a label `road = 1`, `background = 0` to each pixel.

To do so, we have done the following:

* Manipulating the images given to generate more data and better the training of our 'model classifier'. We have handled this process by using various transformation techniques provided by `torchvision.transforms`. In particular we have multiplied the amount of our training images using 18 different transformations. Some of the transformations made are the subsequent: 

    * Resizing
    * Rotation
    * Affine transformation
    * Random cropping
    * Random solarization
    * Color jittering
    * Horizontal flipping

* *The next points to be changed*
* Implementing the 'model'
* Analyzing the models using such figures
* Predicting such results


## Figure representation


## Project structure

The following represents our project structure:

- **[`addittional_data/`](./addittional_data.ipynb)**: Image manipulation to generate an extensive training data set

- **[`model/`](./model.ipynb)**: 'Machine learning model' 'classifier'

- **[`mask_to_submission/`](./mask_to_submission.py)**: Helper function converting from a visualization format to a submission format 

- **[`submission_to_mask/`](./submission_to_mask.py)**: Helper function converting from a submission format to a visualization format 

- **[`Project_Report/`](./CS433_Project2_SFS.pdf)**: Project Report

- **[`README.md`](./README.md)**: This README file


