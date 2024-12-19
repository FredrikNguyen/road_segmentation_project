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


def create_submission():
    """
    Create a submission file called 'submision.csv' from predictions
    """
    foreground_threshold = 0.25 # percentage of pixels = 1 required to assign a foreground label to a patch

    def patch_to_label(patch):
        """
        Assign a label to a patch
        """
        df = np.mean(patch)
        if df > foreground_threshold:
            return 1
        else:
            return 0


    def mask_to_submission_strings(image_filename):
        """
        Reads a single image and outputs the strings that should go into the submission file
        """
        img_number = int(re.search(r"\d+", image_filename).group(0))
        im = mpimg.imread(image_filename)
        im = (im==0).astype(int)
        patch_size = 16
        for j in range(0, im.shape[1], patch_size):
            for i in range(0, im.shape[0], patch_size):
                patch = im[i:i + patch_size, j:j + patch_size]
                label = patch_to_label(patch)
                yield("{:03d}_{}_{},{}".format(img_number, j, i, label))


    def masks_to_submission(submission_filename, *image_filenames):
        """
        Converts images into a submission file
        """
        with open(submission_filename, 'w') as f:
            f.write('id,prediction\n')
            for fn in image_filenames[0:]:
                f.writelines('{}\n'.format(s) for s in mask_to_submission_strings(fn))


    submission_filename = 'submission.csv'
    pred_files = [os.path.join('output', x) for x in os.listdir('output')]
    masks_to_submission(submission_filename, *pred_files)


def visualize_submission(test_dataset):
    """ Visualize submission"""

    label_file = 'submission.csv'

    h = 16
    w = h
    imgwidth = int(math.ceil((600.0/w))*w)
    imgheight = int(math.ceil((600.0/h))*h)

    # Convert an array of binary labels to a uint8
    def binary_to_uint8(img):
        rimg = (img * 255).round().astype(np.uint8)
        return rimg

    def reconstruct_from_labels(image_id):
        im = np.zeros((imgwidth, imgheight), dtype=np.uint8)
        f = open(label_file)
        lines = f.readlines()
        image_id_str = '%.3d_' % image_id
        for i in range(1, len(lines)):
            line = lines[i]
            if not image_id_str in line:
                continue

            tokens = line.split(',')
            id = tokens[0]
            prediction = int(tokens[1])
            tokens = id.split('_')
            i = int(tokens[1])
            j = int(tokens[2])

            je = min(j+w, imgwidth)
            ie = min(i+h, imgheight)
            if prediction == 0:
                adata = np.zeros((w,h))
            else:
                adata = np.ones((w,h))

            im[j:je, i:ie] = binary_to_uint8(adata)

        return im

    random_numbers = random.sample(range(0, len(test_dataset)), 1)
    for i in random_numbers:
        image = test_dataset[i]

        visualize(
            image=denormalize(image.squeeze()),
            #gt_mask=gt_mask[..., 0].squeeze(),
            submission_mask=reconstruct_from_labels(i),
        )