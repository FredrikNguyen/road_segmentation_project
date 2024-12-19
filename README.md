# Road segmentation from satellite imagery using U-net architecture

## About the project

The aim of this project is to train a classifier, using Convolutional Neural Networks (CNN's), and in particular U-nets; that extracts roads in a set of aerial images aquired from GoogleMaps. From each image provided to train the model, there is a ground-truth form of that image where each pixel is labeled as road or background.  We are also provided by a set of images, from where we will test our model by producing our classifier's ground-truth version of those images. Essentially the goal reduces to train our classifier to assign a label `road = 1`, `background = 0` to each pixel. 

To do so, we have done the following:

* Data augmentation to avoid over-fitting and to improve the robustness of the data and better the training of our U-net model. In      particular, performing random image manipulation with a certain probability `p` by making use of the python library `Augmentor`. Some of the transformations made are the subsequent:
    * Left or right rotations of maximum 20 degrees `p = 1`
    * Mirroring images left to right or top to bottom `p = 0.5`
    * Zooming with a 0.8 rate `p = 0.5`
    * Left or right shears of maximum 5 degrees `p = 0.3`
    * Distortions that can make roads appear 'wavy' `p = 0.5`

* Implementing a U-net classifier model, a particular type of CNN and one of the most widely used models for image segmentation. While evaluating the results of our implementation with the real ground truth form of those training images.
* Analyzing the performance of our model through the test images and using various metrics like training and validation loss, dice loss, binary focal loss, combination loss, intersection over union (IOU) score, and F1 score to keep improving our classifier. 
* Predicting perceivable drivable regions from images of GoogleMaps.

## Model evolution representation

The following figure displays the comparison of the predictions obtained with different models for two images from our test image dataset. We can observe the substantial improvement from our first model, to then implementing data augmentation techniques, and finally to our last model.

![Model prediction evolution](https://github.com/CS-433/ml-project-2-sfs_team/tree/6bb9161dfc40055c5fc9ad94c9c96b2fde1d73df/Images/Model_prediction_evolution.png?raw=true)


## Project structure

The following represents our project structure:


- **[`data`](./data/)**: Contains the project's data
  - **[`test`](./data/test/)**: Contains all the images to test our model
  - **[`training`](./data/training/)**: Contains all the data necessary to train our model
    - **[`gorundtruth`](./data/training/groundtruth/)**: Contains the real ground-truth form of the training images
    - **[`images`](./data/training/images/)**: Contains all the images to train our model

- **[`Images`](./Images/)**: Contains any image needed for the project display in GitHub
  - **[`Model_prediction_evolution`](./Images/Model_prediction_evolution.png)** Image used for the README figure representation

- **[`old_model`](./old_model/)**: Contains the old U-net models, and our previous efforts to manipulate the data
  - **[`additional_data`](./old_model/addittional_data.ipynb)**: Previous data manipulation
  - **[`model_keras_unet`](./old_model/model_keras_unet.ipynb)**: First U-net model: Model1 using tensorflow
  - **[`model_pytorch_unet`](./old_model/model_pytorch_unet.ipynb)**: Second U-net model: Model2 using pytorch

- **[`saved_model`](./saved_model/)**: Contains the new U-net model using tensorflow
  - **[`Final_Model`](./saved_model/Final_Model.txt)**: [Kaggle link](https://www.kaggle.com/datasets/fredriknguyenepfl0/saved-model) to download Final Model. We cannot currently upload it in GitHub as it is too heavy. 

- **[`run`](./run.ipynb)**: Sample run of the Final Model. Predicting on the test images, visualizing it, and submitting it

- **[`train`](./train.ipynb)**: Sample train of the Finlal Model. Image manipulation, model training and evaluation 

- **[`utils`](./utils/)**: Contains all the necessary data maniuplation and helper functions that are needed for the saved model
  - **[`data_utils`](./utils/data_utils.py)**: Data augmentation and preprocessing function
  - **[`data_wrappers`](./utils/data_wrappers.py)**: Data loading and cleaning function
  - **[`general_utils`](./utils/general_utils.py)**: Helper functions to plot images and visualize results
  - **[`run_utils`](./utils/run_utils.py)**: Helper functions to plot images and visualize results
  - **[`submission_utils`](./utils/submission_utils.py)**: Helper functions converting from a visualization format to submission format and viceversa





- **[`mask_to_submission`](./mask_to_submission.py)**: Helper function converting from a visualization format to a submission format 

- **[`submission_to_mask`](./submission_to_mask.py)**: Helper function converting from a submission format to a visualization format 

- **[`Project_Report`](./CS433_Project2_SFS.pdf)**: Project Report

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
