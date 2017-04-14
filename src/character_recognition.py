#!/usr/bin/env python3
'''
Predict characters with a sliding window of stepsize=1.
'''

import os
import logging
import sys
from multiprocessing.pool import Pool

import numpy as np
import cv2
from scipy.stats import norm
from skimage import measure

from feature_extraction import get_features_for_window
from character_training import load_model
import config

logging.basicConfig(level=logging.INFO)


def filter_good_characters(texts, layer):
    for text in texts:
        probabilities = text['probabilities'].copy()
        characters=text['characters'].copy()

        # delete zeros (if stepsize > 1)
        probabilities=probabilities[~np.all(probabilities == 0, axis=1)]
        probabilities=probabilities[:,~np.all(probabilities == 0, axis=0)]
        characters=characters[~np.all(characters == '', axis=1)]
        characters=characters[:,~np.all(characters == '', axis=0)]

        for y, x in np.ndindex(*characters.shape):
            if probabilities[y,x] < 0.6:
                characters[y,x] = ''
            if characters[y-1,x] == characters[y,x]:
                characters[y,x] = ''
            if characters[y,x-1] == characters[y,x]:
                characters[y,x] = ''

        # half of a window size gap between characters
        gap_size = 16/config.STEP_SIZE
        text['filtered'] = []

        # make sure there are only text lines with minimum gap of gap_size
        horizontal_character_counts = np.sum(characters!='', axis=1)
        for y in range(characters.shape[0]):

            if max(horizontal_character_counts[max(0,y-gap_size):
                                               min(characters.shape[0]-1,
                                                   y+gap_size)]) == \
                    horizontal_character_counts[y]:
                for character in (characters[y,:]):
                    # TODO add x position and charsize
                    text['filtered'].append({'layer': layer, 'y': y, 'character': character})

                # TODO check horizontal gaps

    return texts



def character_recognition(img, text_probability, dictionary, model,
                          threshold=config.TEXT_RECOGNITION_THRESHOLD):
    '''
    Calculates the texts in an img
    :img: The original image
    :text_probability: The text recognition probability image. pixels
    should be probablities in range [0,1]. has dimension img.shape - (31,31)
    :return: A list of dictionaries, each containing position and text
    '''
    texts = []
    for bbox in bounding_boxes(text_probability,
                               threshold):
        logging.info('Start predicting bbox')
        characters, probabilities = \
            predict_bbox(img, text_probability, bbox, dictionary, model,
                         config.STEP_SIZE,
                         threshold)
        texts.append({'x': bbox[1], 'y': bbox[0],
                      'characters': characters,
                      'probabilities': probabilities})

    return texts


def cut_character(window):
    '''
    :window: A 32x32 window (not preprocessed) containing a character (maybe)
    :return: A 32x32 window with right and left black areas, only containing the
    centered character (ideally)
    '''
    gauss = norm.pdf(list(range(32)), loc=16, scale=2.5)
    gauss = (1-(gauss / gauss.max()))

    threshold1 = 100
    threshold2 = 200

    canny = cv2.Canny(np.uint8(window*255), threshold1, threshold2)
    canny[canny > 0] = 1

    # fig,ax = plt.subplots()
    # ax.imshow(canny)
    # ax.plot(list(range(32)), (32-canny.sum(axis=0))*gauss, color='r')
    # ax.plot(list(range(32)), gauss*32, color='m')

    x1 = ((32-canny.sum(axis=0))*gauss)[:16].argmax()
    x2 = ((32-canny.sum(axis=0))*gauss)[16:].argmax()+16

    window = np.copy(window)
    window[:, :x1] = 0
    window[:, x2:] = 0
    return window


def bounding_boxes(img, threshold):
    # apply threshold:
    blobs = img > threshold
    labeled = measure.label(blobs)

    return [(v.bbox[0], v.bbox[1], v.bbox[2], v.bbox[3])
            for v in measure.regionprops(labeled)]


def bbox_windows(img, text_probability, bbox, model, step_size=1, size=32,
                 threshold=config.TEXT_RECOGNITION_THRESHOLD):
    '''
    Yields all bounding boxes with  high enough text text_probability
    :threshold: defines the necesarry some of probabilties for the
    window to be yielded
    :return: yields (y,x, window) tuples
    '''
    if min([bbox[2]-bbox[0], bbox[3]-bbox[1]]) < 32:
        return

    yields = 0
    misses = 0

    for y in range(bbox[0], bbox[2]-31, step_size):
        for x in range(bbox[1], bbox[3]-31, step_size):
            window = img[y:y+32, x:x+32]
            if text_probability[y:y+32, x:x+32].sum() > threshold*32*32*0.9:
                yields += 1
                yield y-bbox[0], x-bbox[1], window, model
            else:
                misses += 1
    logging.info('Yielded {} and missed {} windows'.format(yields, misses))


def predict_window(args):
    y, x, window, model = args
    try:
        cut_window = cut_character(window)
    except ValueError:
        logging.warn(
            'cut_character failed for position {},{}. window shape {}'.
            format(x, y, window.shape)
        )

        return y, x, '', 0

    features = get_features_for_window(cut_window)
    if features[0]:
        features = features[1].reshape(1, -1)
    else:
        logging.warn(
            'get_features_for_window for position {},{} returned {}.'.
            format(x, y, features[1])
        )
        return y, x, '', 0

    score = model.predict_proba(features).max()

    prediction = model.predict(features)[0]

    return y, x, prediction, score


def predict_bbox(img, text_probability, bbox, dictionary, model, step_size=1,
                 threshold=config.TEXT_RECOGNITION_THRESHOLD):
    '''
    predict all characters in a bbox
    '''
    logging.info('Predicting bbox of size {}x{} at {},{}'.format(
        bbox[2]-bbox[0], bbox[3]-bbox[1], bbox[0], bbox[1]
    ))

    character_probabilities = np.zeros((bbox[2]-bbox[0], bbox[3]-bbox[1]))
    # characters = np.ndarray(shape=(bbox[2]-bbox[0], bbox[3]-bbox[1]),
    #                         dtype='|S1')
    characters = [[''] * (bbox[3]-bbox[1]) for _ in range(bbox[2]-bbox[0])]

    pool = Pool(processes=8)
    i = 0
    for y, x, prediction, score in pool.imap(predict_window,
                                             bbox_windows(img,
                                                          text_probability,
                                                          bbox,
                                                          model,
                                                          step_size,
                                                          threshold=threshold),
                                             8):
        try:
            characters[y][x] = prediction
        except UnicodeEncodeError:
            print('error: {}'.format(prediction))
        else:
            character_probabilities[y, x] = score

        if i % 100 == 0:
            logging.info('{}/{} prediction of window in bbox'.
                         format(i, (bbox[2]-bbox[0]-31)*(bbox[3]-bbox[1]-31)))
        i += 1

    return np.array(characters), character_probabilities

    # TODO now filter the responses..
    # vertical_maxima = \
    #     characters[character_probabilities.argmax(axis=0),
    #                np.array(range(character_probabilities.shape[1]))]
    #
    # return vertical_maxima


if __name__ == "__main__":
    dictionary = np.load(config.DICT_PATH)
    try:
        logging.info('Trying to load model')
        model = load_model()
    except FileNotFoundError:  # noqa
        logging.warn('Model not found, please run character training')
        sys.exit(1)

    filename = os.path.join(config.TEXT_PATH, '2/104.jpg')

    predict_wordfile(filename, model, dictionary)
