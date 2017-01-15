import numpy as np
import pandas as pd
import cv2
import glob
import os
import math
from uuid import uuid4
from preprocessing import preprocess
import config


def get_z(dictionary, x):
    # x should be in dimension 64x1
    flattend_x = np.ndarray.flatten(x)
    tmp = np.abs(np.dot(dictionary.T, flattend_x)) - config.ALPHA
    for i, entry in enumerate(tmp):
        if entry < 0:
            tmp[i] = 0
    return tmp


def get_pooling(z, stepSize, windowSize):
    res = []
    # sliding window over z
    for y in range(0, z.shape[0], stepSize):
        row = []
        for x in range(0, z.shape[1], stepSize):
            pool = z[y:y + windowSize[1], x:x + windowSize[0]]

            # add all entries from pool elementwise
            tmp = np.ones(z.shape[2])
            for i in np.ndindex(pool.shape[:2]):
                tmp += pool[i]

            row.append(tmp)
        res.append(row)
    return np.array(res)


def extract_all_windows(stepSize, windowSize):
    '''
    Return all windows for given image
    '''
    image_files = glob.glob(os.path.join(config.DATASET_PATH, '*.JPG'))

    for f in image_files:

        filename = os.path.splitext(os.path.split(f)[1])[0]

        if not os.path.exists(os.path.join(config.WINDOW_PATH, filename)):
            os.makedirs(os.path.join(config.WINDOW_PATH, filename))

        img = cv2.imread(f)

        for y in range(0, img.shape[0], stepSize):
            for x in range(0, img.shape[1], stepSize):
            # yield the current window
                window = (x, y, img[y:y + windowSize[1], x:x + windowSize[0]])
                cv2.imwrite('{}/{}.JPG'.format(os.path.join(config.WINDOW_PATH, filename), uuid4()), window[2])

    print('finished window extraction.')


def get_features_for_window(dictionary, windowpath):
    """
    return feature representation for given window
    """

    img = cv2.imread(windowpath)
    z = []

    stepSize = 1
    windowSize = (8, 8)

    for y in range(0, img.shape[0] - 7, stepSize):

        row = []

        for x in range(0, img.shape[1] - 7, stepSize):
            # yield the current window
            patch = img[y:y + windowSize[1], x:x + windowSize[0]]

            #preprocess patch
            patch = preprocess(patch)

            #get z entry for preprocessed patch
            row.append(get_z(dictionary, patch))

        #push row to z
        z.append(row)

    #drop most outer lines
    z = np.array(z)[0:-1, 0:-1]

    #pooling
    pooled = get_pooling(z, 8, (8, 8))

    return pooled


def get_features_for_all_imgs(dictionary):
    image_folders = glob.glob(os.path.join(config.WINDOW_PATH, '*'))
    print('creating features for {} images.'.format(len(image_folders)))
    for i, image in enumerate(image_folders):
        window_files = glob.glob(os.path.join(image, '*.JPG'))
        for windowpath in window_files:
            windowname = os.path.splitext(os.path.split(windowpath)[1])[0]
            features = get_features_for_window(dictionary, windowpath)
            np.save(windowpath[:-4], features)
        print('image {} finished'.format(i + 1))
    print('features for {} images created.'.format(len(image_folders)))


if __name__ == "__main__":
    extract_all_windows(32, (32, 32))
    D = np.load(config.DICT_PATH)
    get_features_for_all_imgs(D)
