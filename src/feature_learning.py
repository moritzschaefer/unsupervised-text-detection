"""This code reads in all patches in the patch_directory and builds
a feature representation of it by using a modified version of k means
clustering. The result (the clusters) is saved in a npy file"""

import glob
import logging

import numpy as np

import config
import preprocessing

logging.basicConfig(level=logging.INFO)


def read_files():
    patches = glob.glob('{}/*.npy'.format(config.PATCH_PATH))
    X = np.empty((64, len(patches)))
    for i, patch in enumerate(patches):

        preprocessed_img_patch = preprocessing.preprocess(np.load(patch))

        X[:, i] = preprocessed_img_patch.flatten()

    # check for NaNs
    return X


def init_dictionary():
    D = np.random.rand(64, config.NUM_D)
    D /= np.sqrt(np.sum(D**2, axis=0))
    return D


def find_assignments(X, D):
    # optimize S

    # TODO: probably mini-batch this
    dict_matches = np.dot(X.T, D)
    assignments = np.argmax(dict_matches, axis=1)
    magnitudes = np.max(dict_matches, axis=1)

    return assignments, magnitudes


def average_clusters(X, assignments, magnitudes):
    # optimize D

    D = np.zeros((64, config.NUM_D))
    # sum up all xs in the given row
    for i, x in enumerate(X.T):
        D[:, assignments[i]] += x * magnitudes[i]

    epsilon = 0.00001
    # normalize columns
    # check normalization divisor for 0
    pre = np.sqrt(np.sum(D**2, axis=0))
    pre[pre < epsilon] = epsilon

    D /= pre

    return D


def calc_objective(X, D, assignments, magnitudes):
    '''
    Take the L2 norm of the deviation between X and the clusters and their
    assigned elements in the dataset
    '''

    return np.sum((np.sum(((D[:, assignments] * magnitudes) - X)**2, axis=0)))


def optimize_dictionary(save=True):
    X = read_files()
    D = init_dictionary()
    error = 100000000
    last_error = error + 1

    # optimize until error improvement smaller than the threshold
    while last_error-error > 0.1:
        last_error = error

        assignments, magnitudes = find_assignments(X, D)
        D = average_clusters(X, assignments, magnitudes)
        error = calc_objective(X, D, assignments, magnitudes)

        logging.info('Optimizing dictionary. J: {}'.format(error))

    # save to npy file
    if save:
        np.save(config.DICT_PATH, D)
    return D


if __name__ == "__main__":
    optimize_dictionary()
