"""
main file to run project steps
"""

################################################
# random patch extraction and feature learning #
################################################

import extraction
import config
import xml.etree.ElementTree
import os
import glob
import numpy as np
import feature_learning

print('started random patch extraction and feature learning.')

image_folders = glob.glob(os.path.join(config.TEXT_PATH, '*/'))

for folder in image_folders:

    image_files = glob.glob(os.path.join(folder, '*.jpg'))

    for f in image_files:

        extraction.extract_random_patches(f, config.NUM_PATCHES_PER_TEXT, True)

dictionary = feature_learning.optimize_dictionary()

print('finished random patch extraction and feature learning.')

#####################
# window extraction #
#####################

import xml.etree.ElementTree
import extraction
import config
import glob
import os
import numpy as np

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

import feature_extraction
import extraction
import config
import xml.etree.ElementTree
import os
import glob
import numpy as np

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

from sklearn import svm
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import extraction
import time
import pickle
import glob
import os
import numpy as np

print('started text recognition training.')

text_windows = glob.glob(os.path.join('../data/windows/true/', '*.npy'))
ntext_windows = glob.glob(os.path.join('../data/windows/false/', '*.npy'))

X = []

X = extraction.add_feature_data(text_windows, X)
X = extraction.add_feature_data(ntext_windows, X)

n_text = len(text_windows)
n_ntext = len(ntext_windows)

y = [0] * (n_text + n_ntext)

y[0:n_text] = [1]*n_text

X_shuffled, y_shuffled = shuffle(np.array(X), np.array(y))
X_train, X_test, y_train, y_test = train_test_split(X_shuffled, y_shuffled, test_size=0.20, random_state=7)

svm2 = svm.LinearSVC()
start = time.time()
svm2.fit(X_train, y_train)
end = time.time()
print("Single SVC, training: {t}s, score: {s}".format(t = end - start, s = svm2.score(X_test,y_test)))

filename = 'tr_model.sav'
pickle.dump(svm2, open(filename, 'wb'))

print('finished text recognition training.')

#################
# loading model #
#################
"""
import pickle
from sklearn import svm

loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, y_test)
print(result)
"""
