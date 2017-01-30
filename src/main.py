"""
main file to run project steps
"""

import time
import os
import glob
import numpy as np
import pickle
import xml.etree.ElementTree

import config
import extraction
import feature_learning
import feature_extraction
import text_recognition

################################################
# random patch extraction and feature learning #
################################################


print('started random patch extraction and feature learning.')

"""
for f in image_files:
    extraction.extract_random_patches(f, config.NUM_PATCHES_PER_TEXT, True)

dictionary = feature_learning.optimize_dictionary()

print('finished random patch extraction and feature learning.')
"""
#####################
# window extraction #
#####################


print('started window extraction.')

# get meta info xml
e = xml.etree.ElementTree.parse(config.XML_PATH).getroot()
dic = extraction.parse_xml(e)
n_images = extraction.count_images()
win_per_image = config.TOTAL_WINDOWS_FOR_TRAINING / n_images

image_folders = glob.glob(os.path.join(config.SCENERY_PATH, '*/'))
#print('folder: {}'.format(len(image_folders)))

for folder in image_folders:

    image_files = glob.glob(os.path.join(folder, '*.jpg'))
    #print('images: {}'.format(len(image_files)))

    for f in image_files:
        #print('starting file: {}'.format(f))
        extraction.extract_random_windows(f, 32, (32, 32), int(win_per_image/2), dic, text = True, plot = False)
        extraction.extract_random_windows(f, 32, (32, 32), int(win_per_image/2), dic, text = False, plot = False)


print('finished window extraction.')

######################
# feature extraction #
######################


print('started feature extraction.')

D = np.load(config.DICT_PATH)

text_windows = os.path.join(config.WINDOW_PATH, 'true/')
ntext_windows = os.path.join(config.WINDOW_PATH, 'false/')

feature_extraction.create_features_for_all_windows(text_windows, D)
feature_extraction.create_features_for_all_windows(ntext_windows, D)

print('finished feature extraction.')

#############################
# text recognition training #
#############################

print('started text recognition training.')

X, y = text_recognition.prepare_tr_training_data(text_windows, ntext_windows)

tr_model = text_recognition.train_tr_model(X, y)

text_recognition.save_tr_model(tr_model, config.TEXT_MODEL_PATH)


print('finished text recognition training.')
