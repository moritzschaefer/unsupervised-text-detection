#!/usr/bin/env python3
'''
Predict characters with a sliding window of stepsize=1.
'''

import os
import logging
import sys

import numpy as np
import cv2

from feature_extraction import get_features_for_window
from character_training import load_model, square_patch
import config

    # for (dirpath, dirnames, filenames) in \
    #         os.walk(os.path.join(config.TEXT_PATH)):
    #     for name in filenames:
    #         source_filename = os.path.join(dirpath, name)
    #         img = cv2.imread(source_filename)


def get_predictions(filename, model, dictionary):
    img = cv2.imread(filename)

    # we assume there is just one line of text, so we resize this to 32 pixels
    # height
    scale = 32.0/img.shape[0]
    if scale > 1:
        interpolation = cv2.INTER_LINEAR
    else:
        interpolation = cv2.INTER_AREA

    try:
        resized = cv2.resize(img, None, fx=scale, fy=scale,
                             interpolation=interpolation)
    except Exception as e:
        logging.warning('Error squaring patch: {}'.format(e))
        return []

    extracted_features = []

    target_img = np.ndarray(shape=(32, 32, 3), dtype='uint8')
    for x in range(0, resized.shape[1]-16, 4):
        window = resized[:, x:x+16]

        target_img[:, :, :] = 0
        y_start = int((32-window.shape[0])/2)
        x_start = int((32-window.shape[1])/2)
        target_img[y_start:y_start+window.shape[0],
                    x_start:x_start+window.shape[1],
                    :] = window

        # target_img[:, 0:x_start, :] = \
        #     resized[:, 0, :][:, np.newaxis, :]
        # target_img[:, x_start+resized.shape[1]:, :] = \
        #     resized[:, 0, :][:, np.newaxis, :]
        extracted_features.append(
            get_features_for_window(dictionary, target_img)[1].flatten()
        )

    predictions = model.predict(np.vstack(extracted_features))
    scores = model.decision_function(np.vstack(extracted_features)).max(axis=1)
    return predictions, scores

    # algorithm:
    #
    # - iterate over array at each index do
    # - if character with highest probability changed, set it to the current one
    # set <rising> to True
    # - if probability rises and highest character remains same do nothing
    # - if probability descends AND rising == True, set rising to false, add
    # character to output
    # - if rising = False and it starts to rise again, set rising to true
    import cv2
    import numpy as np
    from matplotlib import pyplot as plt
    img = cv2.imread('messi5.jpg',0)
    edges = cv2.Canny(img,100,200)
    plt.subplot(121),plt.imshow(img,cmap = 'gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(edges,cmap = 'gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    plt.show()

    rising = False
    character = ''
    # max_probabilities = np.max(predictions, axis=0)
    # same =
    #     np.hstack((max_probabilities, [0])) == np.hstack(([0], max_probabilities)) &
    #     np.hstack((max_probabilities, [0,0])) == np.hstack(([0,0], max_probabilities)) &
    #     np.hstack((max_probabilities, [0,0,0])) == np.hstack(([0,0,0], max_probabilities)) &
    #     np.hstack((max_probabilities, [0,0,0,0])) == np.hstack(([0,0,0,0], max_probabilities)) &
    #
    # # for i in range([predictions.shape[0]]):
    # #     if predictions[i]
    #
    # output = []
if __name__ == "__main__":
    dictionary = np.load(config.DICT_PATH)
    try:
        logging.info('Trying to load model')
        model = load_model()
    except FileNotFoundError:  # noqa
        logging.warn('Model not found, please run character training')
        sys.exit(1)

    filename = os.path.join(config.TEXT_PATH, '2/104.jpg')

    get_predictions(filename, model, dictionary)
