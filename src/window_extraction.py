"""
this file extracts windows w or w/o text for training purposes
"""

import time
import os
import glob
import numpy as np
import pickle
import xml.etree.ElementTree

import extraction
import config

def create_windows_for_training():
    # get meta info xml
    e = xml.etree.ElementTree.parse(config.XML_PATH).getroot()
    dic = extraction.parse_xml(e)
    n_images = extraction.count_images()
    win_per_image = config.TOTAL_WINDOWS_FOR_TRAINING / n_images

    image_folders = glob.glob(os.path.join(config.SCENERY_PATH, '*/'))
    #print('folder: {}'.format(len(image_folders)))

    for folder in image_folders:

        image_files = glob.glob(os.path.join(folder, '*.JPG'))
        #print('images: {}'.format(len(image_files)))

        for f in image_files[0:10]:
            #print('starting file: {}'.format(f))
            #extraction.extract_random_windows(f, 32, (32, 32), int(win_per_image/2), dic, text = True, plot = False)
            #extraction.extract_random_windows(f, 32, (32, 32), int(win_per_image/2), dic, text = False, plot = False)
            extraction.extract_random_windows(f, 32, (32, 32), int(5), dic, text = True, plot = False)
            extraction.extract_random_windows(f, 32, (32, 32), int(5), dic, text = False, plot = False)


if __name__ == "__main__":
    create_windows_for_training()
