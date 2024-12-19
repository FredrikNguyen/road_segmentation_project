import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
import re

def imagepath_to_list(im_dir, gt_dir, resize=None):
    """converts list dir to list of array of images

    Args:
        im_dir: path to folder for images
        gt_dir: path to folder for groundtruth
        resize: tuple of integers (height, width)

    Returns:
        list of numpy array representation of the images groundtruth pair"""
    image_files = os.listdir(im_dir)
    image_list = [os.path.join(im_dir, image_files[i]) for i in range(len(image_files))]
    mask_files = os.listdir(gt_dir)
    mask_list = [os.path.join(gt_dir, mask_files[i]) for i in range(len(mask_files))]
    collated_images_and_masks = list(zip(image_list, mask_list))
    if resize:
        images = [[np.asarray(Image.open(y).resize(resize)) for y in x] for x in collated_images_and_masks]
    else: 
        images = [[np.asarray(Image.open(y)) for y in x] for x in collated_images_and_masks]
    
    return images

def plot_random(images):
    """plots image with correspoding groundtruth

    Args:
        images: list of images in numpy format, structured as given by imagepath_to_list()

    Returns:
        None"""
    r_index = random.randint(0, len(images)-1)
    f, axarr = plt.subplots(1, 2)
    axarr[0].imshow(images[r_index][0])
    axarr[1].imshow(images[r_index][1], cmap="gray")
    

def visualize(**images):
    """Helper function for data visualization. Plot images in one row.
    
    Args:
        images: images in numpy format

    Returns:
        None"""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()

def denormalize(x):
    """Helper function for data visualization. Scale image to range 0..1 for correct plot
    Args:
        images: image in numpy format

    Returns:
        None
    """

    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    return x

def extract_number(path):
    """
    Extracts the numerical part of a file name in the format "name_###.png".
    
    Args:
        path (str): The file path or name, expected to include an underscore followed by digits and ending with ".png".
    
    Returns:
        int: The extracted number if found.
        float: Returns `float('inf')` if the pattern is not matched, indicating no valid number was found.
    """
    match = re.search(r"_(\d+)\.png", path)
    return int(match.group(1)) if match else float('inf')

