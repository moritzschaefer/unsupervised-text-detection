"""
this script extracts random patches from images for dictionary training
"""

import os
import glob
import logging

import extraction
import config

logging.basicConfig(level=logging.INFO)

def extract_random_patches_for_training():

    print(extraction.count_images(config.TEXT_PATH))
    print(config.NUM_PATCHES_PER_TEXT)

    n_patches =  extraction.count_images(config.TEXT_PATH) * config.NUM_PATCHES_PER_TEXT

    logging.info('extracting {} patches for dictionary training.'.format(n_patches))

    image_folders = glob.glob(os.path.join(config.TEXT_PATH, '*/'))

    for folder in image_folders:

        image_files = glob.glob(os.path.join(folder, '*.jpg'))

        for f in image_files:

            extraction.extract_random_patches(f, config.NUM_PATCHES_PER_TEXT, config.RESIZE)


if __name__ == "__main__":
    extract_random_patches_for_training()
