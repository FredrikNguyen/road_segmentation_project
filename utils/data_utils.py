import os
from tensorflow import keras
import segmentation_models as sm
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
import Augmentor
from PIL import Image
import random
import albumentations as A
import matplotlib.image as mpimg
import re


def augmentat_images(images, samples):
    """Augments images with rotations, mirroring, zoom, crop, shear, distortions

    Args:
        images: list of images in numpy format, structured as given by imagepath_to_list()
        samples: integer value of number of augmented images
    Returns:
        list of images extended with the augmented images groundtruth pair
        and also list with only augmented pair"""
    p = Augmentor.DataPipeline(images)
    p.rotate(probability=1, max_left_rotation=20, max_right_rotation=20)
    p.flip_left_right(probability=0.5)
    p.zoom_random(probability=0.2, percentage_area=0.8)
    p.flip_top_bottom(probability=0.5)
    p.crop_random(probability=0.2,percentage_area= 0.8)
    p.resize(probability=1,height= 416, width=416)
    p.shear(probability=0.3, max_shear_left=5, max_shear_right=5)
    p.random_distortion(probability=0.3, magnitude=2, grid_height=16, grid_width=16)

    augmented_images = p.sample(samples)
    images.extend(augmented_images)

    return images, augmented_images


def get_augmentation():
    """Add paddings to make image shape divisible by 32
    Args:
        None
    Return:
        None"""
    # This can be extended to perform other augmentation.
    # However, since we already do augmentation before on the data we don't do it here.
    test_transform = [
        A.PadIfNeeded(416, 416)
    ]
    return A.Compose(test_transform)

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        A.Lambda(image=preprocessing_fn),
    ]
    return A.Compose(_transform)