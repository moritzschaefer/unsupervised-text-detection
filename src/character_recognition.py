#!/usr/bin/env python3
'''
Predict characters with a sliding window of stepsize=1.
'''

import os
import logging
import sys

import numpy as np
import cv2
from scipy.stats import norm
from skimage import measure

from feature_extraction import get_features_for_window
from character_training import load_model
import config


def predict_wordfile(filename, model, dictionary):
    '''
    Load a file and predict the whole image as word
    '''
    raise ValueError('Not implemented...')


def character_recognition(img, text_probability, dictionary, model):
    '''
    Calculates the texts in an img
    :img: The original image
    :text_probability: The text recognition probability image. pixels
    should be probablities in range [0,1]
    :return: A list of dictionaries, each containing position and text

    '''
    texts = []
    for bbox in bounding_boxes(text_probability,
                               config.TEXT_RECOGNITION_THRESHOLD):
        text = predict_bbox(img, text_probability, bbox, dictionary, model)
        texts.append({'x': bbox[1], 'y': bbox[0], 'text': text})

    return texts


def cut_character(window):
    '''
    :window: A 32x32 window (not preprocessed) containing a character (maybe)
    :return: A 32x32 window with right and left black areas, only containing the
    centered character (ideally)
    '''
    gauss = norm.pdf(list(range(32)),loc=16,scale=4)
    gauss = (1-(gauss / gauss.max()))

    threshold1 = 100
    threshold2 = 200

    canny = cv2.Canny(window, threshold1, threshold2)
    canny[canny > 0] = 1

    # fig,ax = plt.subplots()
    # ax.imshow(canny)
    # ax.plot(list(range(32)), (32-canny.sum(axis=0))*gauss, color='r')
    # ax.plot(list(range(32)), gauss*32, color='m')

    x1 = ((32-canny.sum(axis=0))*gauss)[:16].argmax()
    x2 = ((32-canny.sum(axis=0))*gauss)[16:].argmax()
    window = np.copy(window)
    window[:x1] = 0
    window[x2:] = 0
    return window


def bounding_boxes(img, threshold):
    # apply threshold:
    blobs = img > threshold
    labeled = measure.label(blobs)

    return [v.bbox for v in measure.regionprops(labeled)]


def bbox_windows(img, text_probability, bbox, size=32,
                 probability_threshold=config.BOUNDING_BOX_THRESHOLD):
    '''
    Yields all bounding boxes with  high enough text text_probability
    :probability_threshold: defines the necesarry some of probabilties for the
    window to be yielded
    :return: yields (y,x, window) tuples
    '''
    if min([bbox[2]-bbox[0], bbox[3]-bbox[1]]) < 32:
        return

    for y in range(bbox[0], bbox[1]-31):
        for x in range(bbox[2], bbox[3]-31):
            window = img[y:y+32, x:x+32]
            if text_probability[y:y+32, x:x+32] > probability_threshold:
                yield y-bbox[0], x-bbox[1], window


def predict_bbox(img, text_probability, bbox, dictionary, model):
    '''
    predict all characters in a bbox
    '''

    character_probabilities = np.zeros((bbox[2]-bbox[0], bbox[3]-bbox[1]))
    characters = np.chararray((bbox[2]-bbox[0], bbox[3]-bbox[1]))

    features = []
    for y, x, window in bbox_windows(img, text_probability, bbox):
        features = get_features_for_window(dictionary,
                                           cut_character(window))[1].flatten()

        character_probabilities[y, x] = model.decision_function(features)
        characters[y, x] = model.predict(features)

    # TODO now filter the responses..
    vertical_maxima = characters[character_probabilities.argmax(axis=0)]

    return vertical_maxima


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
