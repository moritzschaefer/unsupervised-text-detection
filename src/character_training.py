#!/usr/bin/env python3
import os
import pickle
import xml.etree.ElementTree as ET
import logging

import cv2
import numpy as np
from sklearn.svm import LinearSVC
import sklearn
import matplotlib.pyplot as plt

from feature_extraction import get_features_for_window
from plot_confusion_matrix import plot_confusion_matrix
import config

logging.basicConfig(level=logging.INFO)


def square_patches(path, target):
    '''
    Converts all images to 32x32 patches and moves them to the target directory
    '''
    target_img = np.ndarray(shape=(32, 32, 3), dtype='uint8')
    for (dirpath, dirnames, filenames) in os.walk(os.path.join(path, 'char')):
        for name in filenames:
            source_filename = os.path.join(dirpath, name)

            img = cv2.imread(source_filename)
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
                continue

            # copy into target_img
            target_img[:, :, :] = 0
            y_start = int((32-resized.shape[0])/2)
            x_start = int((32-resized.shape[1])/2)
            target_img[y_start:y_start+resized.shape[0],
                       x_start:x_start+resized.shape[1],
                       :] = resized

            # extend the empty areas at the sides with the
            if y_start > 0:
                target_img[0:y_start, :, :] = resized[0, :, :][np.newaxis, :, :]
                target_img[y_start+resized.shape[0]:, :, :] = \
                    resized[0, :, :][np.newaxis, :, :]
            if x_start > 0:
                target_img[:, 0:x_start, :] = resized[:, 0, :][:, np.newaxis, :]
                target_img[:, x_start+resized.shape[1]:, :] = \
                    resized[:, 0, :][:, np.newaxis, :]
            os.makedirs(os.path.join(target, os.path.relpath(dirpath, path)),
                        exist_ok=True)

            cv2.imwrite(os.path.join(target,
                                     os.path.relpath(dirpath, path),
                                     name),
                        target_img)


def create_data_set(dir, labels, dictionary):
    tree = ET.parse(labels)
    labels = []
    features = []
    for child in tree.getroot():
        filename = child.attrib['file']
        try:
            extracted_features = extract_feature_vector(os.path.join(dir,
                                                                     filename),
                                                        dictionary)
        except Exception as e:
            import ipdb
            ipdb.set_trace()
            logging.warn('Could not find file {}. Skip')
        else:
            labels.append(child.attrib['tag'])
            features.append(extracted_features[1].flatten())

    # np.savetxt(FEATURE_FILE, np.array(features))
    # with open(LABEL_FILE, 'w') as f:
    #     f.writelines(labels)

    return features, labels


def extract_feature_vector(filename, dictionary):
    try:
        return get_features_for_window(dictionary, filename)
    except ValueError as e:
        print('file {} couldn\'t be read: {}'.format(filename, e.message))


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
    import ipdb
    ipdb.set_trace()
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
    features, labels = create_data_set(
        os.path.join(config.DATA_DIR, 'character_icdar_train/extracted/'),
        os.path.join(config.DATA_DIR, 'character_icdar_train/char.xml'),
        dictionary)
    logging.info('Created training data set. Now training SVM')
    model = train_character_svm(features, labels)
    logging.info('Trained model')
    return model


if __name__ == "__main__":

    dictionary = np.load(config.DICT_PATH)
    try:
        logging.info('Trying to load model')
        model = load_model()
    except FileNotFoundError:  # noqa
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
            os.path.join(config.DATA_DIR, 'character_icdar_test/char.xml'),
            dictionary)
        logging.info('Test data loaded. Predicting test data')
        with open('test_set.pkl', 'wb') as f:
            pickle.dump((test_features, test_labels), f)

    label_set = np.unique(test_labels)
    predicted_labels = model.predict(test_features)

    c_matrix = sklearn.metrics.confusion_matrix(test_labels,
                                                predicted_labels,
                                                label_set)

    logging.info('Printing confusio matrix')
    print(c_matrix)

    # plot
    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(c_matrix, classes=label_set,
                          title='Confusion matrix, without normalization')
    import ipdb
    ipdb.set_trace()
    plt.show()
