"""feature extraction for windows"""

from multiprocessing import Pool
import numpy as np
import cv2
import glob
import os
from uuid import uuid4
from preprocessing import preprocess
import config
import logging

logging.basicConfig(level=logging.INFO)

def get_z(x, dictionary):

    # x should be in dimension 64x1
    flattend_x = np.ndarray.flatten(x)

    # get value |D^Tx| - ALPHA
    tmp = np.abs(np.dot(dictionary.T, flattend_x)) - config.ALPHA

    # choose maximum from 0 and tmp
    indexes = np.where(tmp < float(0))
    tmp[indexes] = 0

    return tmp

def get_pooling(z):
    #import pdb; pdb.set_trace()
    res = []
    # sliding window over z, no overlap
    for y in range(0, z.shape[0], 8):
        row = []
        for x in range(0, z.shape[1], 8):
            pool = z[x:x + 8, y:y + 8]
            #print(pool)

            # add all entries from pool elementwise
            tmp = np.zeros(z.shape[2])

            # iterate over last dimension
            for i in np.ndindex(pool.shape[:2]):
                tmp += pool[i]

            row.append(tmp)
        res.append(row)
    return np.array(res)

def get_features_for_window(window):
    """
    :window: either an np.ndarray or a filename to a window
    return feature representation for given window
    """
    dictionary = np.load(config.DICT_PATH)

    if not isinstance(window, (np.ndarray, np.generic)):
        img = np.load(window)
    else:
        img = window

    if img is None or img.shape != (32, 32, 3):
        #raise ValueError('Image doesn\'t exist or is not a 32x32 patch')
        return (False, np.array(img).shape)

    z = []
    # sliding window over window
    for y in range(0, img.shape[0] - 7):

        row = []

        for x in range(0, img.shape[1] - 7):
            # yield the current window
            patch = img[x:x + 8, y:y + 8]

            # preprocess patch
            patch = preprocess(patch)

            #get z entry for preprocessed patch
            row.append(get_z(patch, dictionary))

        #push row to z
        z.append(row)

    z = np.array(z)

    if z.shape != (25, 25, config.NUM_D):
        return (False, z.shape)

    # drop most outer lines
    z = z[0:-1, 0:-1]

    #pooling
    pooled = get_pooling(z)

    return (True, pooled, window)


def create_features_for_all_windows(path, text, n_jobs=1):

    window_files = glob.glob(os.path.join(path, '*.npy'))
    logging.info('creating features for {} windows.'.format(len(window_files)))

    if text:
        save_path = os.path.join(config.FEATURE_PATH, 'true/')
    else:
        save_path = os.path.join(config.FEATURE_PATH, 'false/')

    # multiprocessing
    p = Pool(n_jobs)
    results = p.map(get_features_for_window, window_files)

    p.close()
    p.join()

    # count wrong computations
    counter = 0
    for i, features in enumerate(results):
        # feature computation succesful
        if features[0]:
            np.save(os.path.join(save_path, os.path.basename(features[2])), features[1])
        else:
            # computation not succesful
            counter += 1

    logging.info('finished feature extraction. {} malformed windows encounterd.'.format(counter))


if __name__ == "__main__":
    D = np.load(config.DICT_PATH)

    text_windows = os.path.join(config.WINDOW_PATH, 'true/')
    ntext_windows = os.path.join(config.WINDOW_PATH, 'false/')

    create_features_for_all_windows(text_windows, True, n_jobs=8)
    create_features_for_all_windows(ntext_windows, False, n_jobs=8)
