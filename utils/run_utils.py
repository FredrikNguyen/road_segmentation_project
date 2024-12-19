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
import math
from PIL import Image

from utils.general_utils import *

def create_predictions(model ,test_dataset):
    """
    creates directory where predictions are stored
    Args:
        model: trained model
        test_dataset: test_dataset
    
    Returns:
        None
    
    """
    directory_name = "output"
    try:
        os.mkdir(directory_name)
        print(f"Directory '{directory_name}' created successfully.")
    except FileExistsError:
        pass

    for idx, image in enumerate(test_dataset):
        image = np.expand_dims(image, axis=0)
        pr_mask = model.predict(image)
        pr_mask = (pr_mask > 0.5).astype(int)*255
        mask_filename = f"pred_{1+idx}.png"
        mask_path = os.path.join(directory_name, mask_filename)
        cv2.imwrite(mask_path, pr_mask.squeeze())

def visualize_predict(model, test_dataset):
    """
    Visualize a prediction
    
    Args:
        model: trained model
        test_dataset: test_dataset
        nb_samples: integer value 
    
    Returns:
        None
    """
    random_numbers = random.sample(range(0, len(test_dataset)), 1)
    for i in random_numbers:
        image = test_dataset[i]
        image = np.expand_dims(image, axis=0)
        pr_mask = model.predict(image)
        pr_mask = (pr_mask > 0.5).astype(int)*255

    visualize(
        image=denormalize(image.squeeze()),
        #gt_mask=gt_mask[..., 0].squeeze(),
        pr_mask=pr_mask[..., 0].squeeze(),
    )



def plot_metrics(history):
    """
    Plots metrics and loss from model training such as f1-score, loss and iou_score
    """
    # Plot training & validation iou_score values
    plt.figure(figsize=(30, 5))
    plt.subplot(131)
    plt.plot(history.history['iou_score'])
    plt.plot(history.history['val_iou_score'])
    plt.title('Model iou_score')
    plt.ylabel('iou_score')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(132)
    plt.plot(history.history['f1-score'])
    plt.plot(history.history['val_f1-score'])
    plt.title('Model f1-score')
    plt.ylabel('f1-score')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(133)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('loss.png')
    plt.show()