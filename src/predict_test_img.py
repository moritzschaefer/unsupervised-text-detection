#!/usr/bin/env python3

'''
This is the central main file combining all the prediction functionality.
It fetches images from config.TEST_IMAGE_PATH and predicts the parts with text
contained and the characters contained.
TODO: integrate better character_recognition from notebook
TODO2: save outputs as: for each input image
  - One json containing coordinates of text regions and recognized characters
  (along with positions and sizes)
  - The original image (png) overlayed with bounding boxes of detected text and
  printed recognized characters
  - npy of text detection
  - npy of character recognition
'''

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
from character_recognition import character_recognition, filter_good_characters

logging.basicConfig(level=logging.INFO)


def sliding_window(img, model, step_size=1):
    '''
    Yield all windows in  an image
    '''
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
    except Exception as e:
        print(x, y, e)
        return x, y, 0
    return x, y, v[0][1]


def get_prediction_values(img, model, step_size=1):
    '''
    Calculate the text_recognition probabilities for each pixel for each
    layer
    :return: A list of tuples (img with layer dimensions, prediction values)
    '''
    layers = []
    for i, layer_img in enumerate(get_all_layers(img)):
        pool = Pool(processes=8)
        padded_img = cv2.copyMakeBorder(layer_img, 32, 32, 32, 32,
                                        cv2.BORDER_REFLECT)
        values = np.zeros(shape=[padded_img.shape[0], padded_img.shape[1]],
                          dtype='float')
        pixel_counter = 0
        logging.info('started for layer {}'.format(i))
        for x, y, v in pool.imap(async_predict,
                                 sliding_window(padded_img.astype('float32'),
                                                model,
                                                step_size),
                                 8):
            values[y:y+step_size+32, x:x+step_size+32] += v

            if (pixel_counter) % 100 == 0:
                logging.info("pixel_counter: {}/{} from layer {}".
                             format(pixel_counter,
                                    ((padded_img.shape[0] - 31) *
                                     (padded_img.shape[1] - 31)) //
                                    step_size**2,
                                    i))
            pixel_counter += 1
        pool.close()
        pool.join()

        layers.append((layer_img.astype('float32')[32:-32,32:-32], values))
        logging.info('finished layer {}'.format(i))
    return layers


# return all Scaling image of a Image,save into Layer Matrix
def get_all_layers(img):
    for (i, resized) in enumerate(
        pyramid_gaussian(img,
                         downscale=config.LAYER_DOWNSCALE,
                         max_layer=config.NUM_LAYERS)):
        # if the image is too small, break from the loop
        if resized.shape[0] < 32 or resized.shape[1] < 32:
            break
        yield resized


def predict_images(step_size=1, plot=True, character=True):
    '''
    Predict all images in config.TEST_IMAGE_PATH
    Save the predictions in TEST_IMAGE_PATH
    '''
    text_model = pickle.load(open(config.TEXT_MODEL_PATH, 'rb'))  # get model
    if character:
        character_model = load_model()
    dictionary = np.load(config.DICT_PATH)  # get dictionary
    image_files = glob.glob(os.path.join(config.TEST_IMAGE_PATH, '*.png'))

    for filename in image_files:
        img = cv2.imread(filename)
        logging.info('started computation for img {}'.
                     format(filename.split('/')[-1].split('.')[0]))

        predicted_layers = get_prediction_values(img, text_model, step_size)
        texts = []
        for layer, (layer_img, layer_predictions) in enumerate(predicted_layers):
            # compute
            if plot:
                cv2.imshow("image 1", layer_img)
                cv2.imshow("image 2", layer_predictions/layer_predictions.max())
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            np.save('{}/{}_layer_{}_prediction.npy'.format(
                config.TEST_IMAGE_PATH,
                filename.split('/')[-1].split('.')[0], layer),
                layer_predictions)

            if character:
                print('Calculate Characters for layer {}'.format(
                    layer_img.shape))
                layer_texts = character_recognition(layer_img,
                                                    layer_predictions,
                                                    dictionary,
                                                    character_model)
                texts.extend(filter_good_characters(layer_texts, layer))
        if texts:
            pickle.dump(texts, open('{}/{}_character_predictions.pkl'.format(
                config.TEST_IMAGE_PATH,
                filename.split('/')[-1].split('.')[0]), 'w'))

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
    predict_images(config.STEP_SIZE, plot=False, character=True)
