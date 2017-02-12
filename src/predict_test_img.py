#!/usr/bin/env python3

import os
from multiprocessing.pool import Pool
import pickle
import logging
import glob

import cv2
import numpy as np
from skimage.transform import pyramid_gaussian

from character_training import load_model
import feature_extraction
import config
from character_recognition import character_recognition

logging.basicConfig(level=logging.INFO)

# get all windows of a image
def sliding_window(img, model, step_size=1):
    for y in range(0, img.shape[0]-32, step_size):
        for x in range(0, img.shape[1]-32, step_size):
            yield (x,
                   y,
                   img[y:min(y+32, img.shape[0]),
                       x:min(x+32, img.shape[1]),
                       :],
                   model)


def async_predict(args):
    x, y, window, model = args
    features = feature_extraction.\
        get_features_for_window(window)
    # reshape it so it contains a single sample
    try:
        v = model.predict_proba(features[1].flatten().reshape(1, -1))
    except:
        pass
    return x, y, v[0][1]


def get_prediction_values(img, model, step_size=1):
    '''
    Calculate the text text_recognition probabilities for each pixel for each
    layer
    '''
    layers = []
    for i, layer_img in enumerate(get_all_layers(img)):
        pool = Pool(processes=8)
        values = np.zeros(shape=[layer_img.shape[0], layer_img.shape[1]],
                          dtype='float')
        pixel_counter = 0
        logging.info('started for layer {}'.format(i))
        for x, y, v in pool.imap(async_predict,
                                 sliding_window(layer_img.astype('float32'),
                                                model,
                                                step_size),
                                 8):
            values[y:y+step_size+32, x:x+step_size+32] += v

            if (pixel_counter) % 100 == 0:
                logging.info("pixel_counter: {}/{} from layer {}".
                format(pixel_counter,
                       ((layer_img.shape[0] - 31) *(layer_img.shape[1] - 31))//step_size**2,
                       i))
            pixel_counter += 1
        pool.close()
        pool.join()

        layers.append((layer_img.astype('float32'), values))
        logging.info('finished layer {}'.format(i))
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


def predict_images(step_size=1, plot=True, character=True):
    text_model = pickle.load(open(config.TEXT_MODEL_PATH, 'rb'))  # get model
    if character:
        character_model = load_model()
    dictionary = np.load(config.DICT_PATH)  # get dictionary
    image_files = glob.glob(os.path.join(config.TEST_IMAGE_PATH + '/test_set/', '*.png'))
    #image_files = [os.path.join(config.TEST_IMAGE_PATH, 'test3.png')]

    for filename in image_files:
        img = cv2.imread(filename)
        logging.info('started computation for img {}'.format(filename.split('/')[-1].split('.')[0]))

        predicted_layers = get_prediction_values(img, text_model, step_size)
        for layer_img, layer_predictions in predicted_layers:
            # compute
            if plot:
                cv2.imshow("image 1", layer_img)
                cv2.imshow("image 2", layer_predictions/layer_predictions.max())
                cv2.waitKey(0)
                cv2.destroyAllWindows()


            np.save('../data/test_images/test_set/{}_prediction.npy'.format(
                            filename.split('/')[-1].split('.')[0]),
                                                layer_predictions)

            if character:
                print('Calculate Characters for layer {}'.format(
                                          layer_img.shape))
                texts = character_recognition(layer_img, layer_predictions,
                                          dictionary, character_model)

                print(texts)
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
    predict_images(config.STEP_SIZE, plot=False, character=False)
