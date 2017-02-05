#!/usr/bin/env python3

import os
from multiprocessing.pool import Pool
import pickle

import cv2
import numpy as np
from skimage.transform import pyramid_gaussian

from character_training import load_model
import feature_extraction
import config
from character_recognition import character_recognition


# get all windows of a image
def sliding_window(img, model, step_size=1):
    for y in range(0, img.shape[0]-32, step_size):
        for x in range(0, img.shape[1]-32, step_size):
            yield (x,
                   y,
                   img[y:min(y+32, img.shape[0]), x:min(x+32, img.shape[1]), :],
                   model)


def async_predict(args):
    x, y, window, model = args
    print(x, y)
    features = feature_extraction.\
        get_features_for_window(window.astype('float32'))
    # reshape it so it contains a single sample
    v = model.predict_proba(features[1].flatten().reshape(1, -1))
    return x, y, v[0][1]


def get_prediction_values(img, model):
    '''
    Calculate the text text_recognition probabilities for each pixel for each
    layer
    '''
    layers = []
    for layer_img in get_all_layers(img):
        pool = Pool(processes=8)
        values = np.zeros(shape=[img.shape[0]-31, img.shape[1]-31],
                          dtype='float')

        for x, y, v in pool.imap(async_predict,
                                 sliding_window(layer_img, model, 8), 8):
            print(v)
            values[y, x] = v

        pool.close()
        pool.join()

        layers.append((layer_img, values))
    return layers


# return all Scaling image of a Image,save into Layer Matrix
def get_all_layers(img):
    for (i, resized) in enumerate(pyramid_gaussian(img,
                                                   downscale=1.3,
                                                   max_layer=0)):  # TODO use 7
        # if the image is too small, break from the loop
        if resized.shape[0] < 32 or resized.shape[1] < 32:
            break
        yield resized


def predict_images():
    text_model = pickle.load(open(config.TEXT_MODEL_PATH, 'rb'))  # get model
    character_model = load_model()
    dictionary = np.load(config.DICT_PATH)  # get dictionary
    # image_files = glob.glob(os.path.join(config.TEST_IMAGE_PATH, '*.jpg'))
    image_files = [os.path.join(config.TEST_IMAGE_PATH, '111-1137_IMG.jpg')]

    for filename in image_files:
        img = cv2.imread(filename)
        predicted_layers = get_prediction_values(img, text_model)
        for layer_img, layer_predictions in predicted_layers:
            # compute
            print('Calculate Characters for layer {}'.format(layer_img.shape))
            texts = character_recognition(layer_img, layer_predictions,
                                          dictionary, character_model)

            print(texts)
            cv2.imshow("image 1", layer_img)
            cv2.imshow("image 2", layer_predictions)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


        # combine_probability_layers(img, predicted_layers)


def combine_probability_layers(img, layers):
    '''
    Return a combined image of all probabilities
    '''
    text_probability_image = np.zeros(img.shape, float)

    for y in range(0, img.shape[0]):
        for x in range(0, img.shape[1]):
            max_probability = 0
            for layer in layers:
                # x and y in the layer which correspond to position in
                # original image
                trans_y = (layer.shape[0]/img.shape[0]) * y
                trans_x = (layer.shape[1]/img.shape[1]) * y

                window = layer[max(0, trans_y-32):
                               min(trans_y+1, layer.shape[0]),
                               max(0, trans_x-32):
                               min(trans_x+1, layer.shape[1])]

                max_probability = max(max_probability, window.max())

            text_probability_image[y, x] = max_probability

    cv2.imshow("image 1", img)
    cv2.imshow("image 2", text_probability_image)
    cv2.waitKey(0)

    return text_probability_image

if __name__ == "__main__":
    predict_images()
