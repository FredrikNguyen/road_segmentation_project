# Road segmentation from satellite images using 'machine learning techniques'

## About the Project

The aim of this project is to train a 'model classifier' that extracts roads in a set of aerial images aquired from GoogleMaps. From each image provided to train the model, there is a ground-truth form of that image where each pixel is labeled as road or background.  We are also provided by a set of images, from where we will test our model by producing our classifier's ground-truth version of those images. Essentially the goal reduces to train a classifier to assign a label `road = 1`, `background = 0` to each pixel.

To do so, we have done the following:

* Manipulating the images given to generate more data and better the training of our 'model classifier'. We have handled this process by using various transformation techniques provided by `torchvision.transforms`. In particular we have multiplied our training images using 18 different transformations. Some of the transformations made are the subsequent: 

 * Resizing
 * Rotation
 * Affine transformation
 * Random cropping
 * Random solarization
 * Color jittering
 * Horizontal flipping

* 
*