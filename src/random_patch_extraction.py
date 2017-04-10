"""
Extract random patches from images for dictionary training
"""

import os
import glob
import logging
import cv2
import numpy as np
import random
from uuid import uuid4

import config

logging.basicConfig(level=logging.INFO)


def count_images(path):
    """
    counts images
    """
    c = 0

    image_folders = glob.glob(os.path.join(path, '*/'))

    for folder in image_folders:
        image_files = glob.glob(os.path.join(folder, '*.jpg'))
        c += len(image_files)
    return c


def extract_random_patches(path, patches, resize=True):
    '''
    Return random patches for img in path
    PATCHES ARE NOT PREPROCESSED!
    '''

    img = cv2.imread(path)

    if resize:
        height, width = img.shape[:2]
        img = cv2.resize(img, (int(32*width/height), 32),
                         interpolation=cv2.INTER_AREA)

    for i in range(patches):
        x = random.randint(0, img.shape[1] - 8)
        y = random.randint(0, img.shape[0] - 8)
        patch = img[y:y+8, x:x+8]
        # write patch to fs
        np.save('{}/{}.npy'.format(config.PATCH_PATH, uuid4()), patch)


def extract_random_patches_for_training():
    n_patches = count_images(config.TEXT_PATH) * config.NUM_PATCHES_PER_TEXT

    logging.info('extracting {} patches for dictionary training.'.
                 format(n_patches))

    image_folders = glob.glob(os.path.join(config.TEXT_PATH, '*/'))

    for folder in image_folders:
        image_files = glob.glob(os.path.join(folder, '*.jpg'))

        for f in image_files:
            extract_random_patches(f, config.NUM_PATCHES_PER_TEXT,
                                   config.RESIZE_WORDS_FOR_DIC)


if __name__ == "__main__":
    extract_random_patches_for_training()
