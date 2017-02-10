#!/usr/bin/env python3
import os
import pickle
import xml.etree.ElementTree as ET
import logging
import random
import sys

import cv2
import numpy as np
from sklearn.svm import LinearSVC
import sklearn

from feature_extraction import get_features_for_window
from plot_confusion_matrix import plot_confusion_matrix
import config

logging.basicConfig(level=logging.INFO)


def square_patch(img, mode='black'):
    '''
    mode can be 'black', 'fill' and 'random'
    TODO random not working cause of missing last_image. maybe just disable that
    option
    '''
    target_img = np.zeros(shape=(32, 32, 3), dtype='uint8')
    scale = 32.0/max(img.shape[:2])
    if scale > 1:
        interpolation = cv2.INTER_LINEAR
    else:
        interpolation = cv2.INTER_AREA

    try:
        resized = cv2.resize(img, None, fx=scale, fy=scale,
                             interpolation=interpolation)
    except Exception as e:
        logging.warning('Error squaring patch: {}'.format(e))
        raise

    y_start = int((32-resized.shape[0])/2)
    x_start = int((32-resized.shape[1])/2)
    target_img[y_start:y_start+resized.shape[0],
               x_start:x_start+resized.shape[1],
               :] = resized

    # extend the empty areas at top and bottom with repetition
    if mode == 'fill':
        if y_start > 0:
            target_img[0:y_start, :, :] = resized[0, :, :][np.newaxis, :, :]
            target_img[y_start+resized.shape[0]:, :, :] = \
                resized[0, :, :][np.newaxis, :, :]

    # extend the empty areas at the sides with random characters
    if x_start > 0:
        last_image = None
        if mode == 'random' and last_image != None:  # noqa
            # use last character (as they are not sorted, it's OK)
            try:
                start_index = random.randint(0, last_image.shape[1]-x_start-1)
            except:
                start_index = 0

            target_img[:, 0:x_start, :] = \
                last_image[:, start_index:start_index+x_start, :]
            missing_pixels = 32-(x_start+resized.shape[1])
            target_img[:, x_start+resized.shape[1]:, :] = \
                last_image[:, start_index:start_index+missing_pixels, :]
        elif mode == 'fill':
            target_img[:, 0:x_start, :] = \
                resized[:, 0, :][:, np.newaxis, :]
            target_img[:, x_start+resized.shape[1]:, :] = \
                resized[:, 0, :][:, np.newaxis, :]

    # fill characters shouldnt be too small
    if resized.shape[0] == 32 and resized.shape[1] >= 14:
        last_image = resized
    return target_img


def square_patches(path, target):
    '''
    Converts all images to 32x32 patches and moves them to the target directory
    '''
    for (dirpath, dirnames, filenames) in os.walk(os.path.join(path, 'char')):
        for name in filenames:
            source_filename = os.path.join(dirpath, name)

            img = cv2.imread(source_filename)

            os.makedirs(os.path.join(target, os.path.relpath(dirpath, path)),
                        exist_ok=True)
            try:
                cv2.imwrite(os.path.join(target,
                                         os.path.relpath(dirpath, path),
                                         name),
                            square_patch(img))
            except Exception as e:
                logging.warn('Exception for {}: {}'.format(name, e))


def create_data_set(dir, label_file):
    tree = ET.parse(label_file)
    labels = []
    features = []
    for child in tree.getroot():
        filename = child.attrib['file']
        try:
            extracted_features = extract_feature_vector(os.path.join(dir,
                                                                     filename))
            if not extracted_features[0]:
                import ipdb
                ipdb.set_trace()
                continue

        except FileNotFoundError:  # noqa
            logging.warn('Could not find file {}. Skip'.format(filename))
            raise
        except Exception as e:
            logging.warn('error: {}'.format(e))
            raise
        else:
            labels.append(child.attrib['tag'])
            features.append(extracted_features[1].flatten())

    # np.savetxt(FEATURE_FILE, np.array(features))
    # with open(LABEL_FILE, 'w') as f:
    #     f.writelines(labels)

    return features, labels


def extract_feature_vector(filename):
    try:
        return get_features_for_window(filename)
    except ValueError as e:
        print('file {} couldn\'t be read: {}'.format(filename, e))
        raise


def train_character_svm(features, labels):
    model = LinearSVC(penalty='l2',
                      loss='squared_hinge',
                      dual=True,
                      tol=0.0001,
                      C=1.0,
                      multi_class='ovr',
                      fit_intercept=True,
                      intercept_scaling=1,
                      class_weight=None,
                      verbose=0,
                      random_state=None,
                      max_iter=1000)
    # check features.shape = [n_samples, n_features]
    model.fit(features, labels)
    return model


def load_model():
    with open(config.CHARACTER_MODEL_PATH, 'rb') as f:
        return pickle.load(f)


def _save_model(model):
    with open(config.CHARACTER_MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    logging.info('Model saved in {}'.format(config.CHARACTER_MODEL_PATH))


def train_model():
    # create training set and fit model
    square_patches(os.path.join(config.DATA_DIR, 'character_icdar_train/'),
                   os.path.join(config.DATA_DIR,
                                'character_icdar_train/extracted/'))
    logging.info('Created square patches. Extracting training data set')
    features, labels = create_data_set(
        os.path.join(config.DATA_DIR, 'character_icdar_train/extracted/'),
        os.path.join(config.DATA_DIR, 'character_icdar_train/char.xml'))
    logging.info('Created training data set. Now training SVM')
    model = train_character_svm(features, labels)
    logging.info('Trained model')
    return model


if __name__ == "__main__":

    try:
        logging.info('Trying to load model')
        model = load_model()
    except (FileNotFoundError, Exception):  # noqa
        logging.info('Model not found. Training model...')
        model = train_model()
        logging.info('Saving model')
        _save_model(model)

    try:
        with open('test_set.pkl', 'rb') as f:
            test_features, test_labels = pickle.load(f)
    except FileNotFoundError:  # noqa
        # now apply the test set
        logging.info('Creating squared test patches')
        square_patches(
            os.path.join(config.DATA_DIR, 'character_icdar_test'),
            os.path.join(config.DATA_DIR, 'character_icdar_test/extracted'))
        logging.info('Building test data set')
        test_features, test_labels = create_data_set(
            os.path.join(config.DATA_DIR, 'character_icdar_test/extracted'),
            os.path.join(config.DATA_DIR, 'character_icdar_test/char.xml'))
        logging.info('Test data loaded. Predicting test data')
        with open('test_set.pkl', 'wb') as f:
            pickle.dump((test_features, test_labels), f)

    label_set = np.unique(test_labels)
    predicted_labels = model.predict(test_features)

    c_matrix = sklearn.metrics.confusion_matrix(test_labels,
                                                predicted_labels,
                                                label_set)

    logging.info('Printing confusion matrix')
    print(c_matrix)

    # plot
    # Plot non-normalized confusion matrix
    if len(sys.argv) > 1:
        import matplotlib.pyplot as plt
        plt.figure()
        plot_confusion_matrix(c_matrix, classes=label_set,
                            title='Confusion matrix, without normalization')
        plt.show()
