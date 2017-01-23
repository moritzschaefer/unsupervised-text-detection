"""This code reads in all patches in the patch_directory and builds
a feature representation of it by using a modified version of k means
clustering. The result (the clusters) is saved in a npy file"""

import glob
import logging

import numpy as np
import cv2

import config

logging.basicConfig(level=logging.INFO)


def read_files():
    patches = glob.glob('{}/*.JPG'.format(config.PATCH_PATH))
    X = np.empty((64, len(patches)))
    for i, patch in enumerate(patches):

        X[:, i] = cv2.imread(patch, 0).flatten()

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

    # normalize columns
    D /= np.sqrt(np.sum(D**2, axis=0))

    return D


def calc_objective(X, D, assignments, magnitudes):

    # THIS DOESN'T WORK (L1 norms squared summed)
    # 1. differences of clusters to Xes
    # 2. L1-norm for each datapoint
    # 3. square the norms
    # 4. sum the squares

    # L2 norm sums work!
    return np.sum((np.sum(((D[:, assignments] * magnitudes) - X)**2, axis=0)))


def optimize_dictionary(save=True):
    X = read_files()
    D = init_dictionary()
    error = 100000000
    last_error = error + 1

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
