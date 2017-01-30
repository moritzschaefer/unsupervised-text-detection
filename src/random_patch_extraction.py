"""
this script extracts random patches from images for dictionary training
"""

import os
import glob

import extraction
import config


def extract_random_patches_for_training():
    image_folders = glob.glob(os.path.join(config.TEXT_PATH, '*/'))

    for folder in image_folders:

        image_files = glob.glob(os.path.join(folder, '*.jpg'))

        for f in image_files[0:2]:

            extraction.extract_random_patches(f, config.NUM_PATCHES_PER_TEXT, False)


if __name__ == "__main__":
    print('yeah')
    extract_random_patches_for_training()
