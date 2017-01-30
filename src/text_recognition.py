"""
this codes trains, saves and loads a text recognition model
"""
import pickle
import glob
import os
import time

import numpy as np
import config

from sklearn import svm
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# load previously build model
def load_tr_model(path):
    loaded_model = pickle.load(open(path, 'rb'))
    return loaded_model


# save model
def save_tr_model(model, filename):
    pickle.dump(model, open(filename, 'wb'))


def add_feature_data(windowspath, X=[]):
    """
    adds feature representations from windowspath to dataset
    """
    for window in windowspath:
        w = np.load(window)
        X.append(np.array(w).flatten())
    return X


# prepare data
def prepare_tr_training_data(text_windows_path, n_text_windows_path):
    """

    """
    text_windows = glob.glob(os.path.join(text_windows_path, '*.npy'))
    ntext_windows = glob.glob(os.path.join(n_text_windows_path, '*.npy'))

    X = []

    X = add_feature_data(text_windows, X)
    X = add_feature_data(ntext_windows, X)

    n_text = len(text_windows)
    n_ntext = len(ntext_windows)

    y = [0] * (n_text + n_ntext)

    y[0:n_text] = [1]*n_text

    return shuffle(np.array(X), np.array(y))


# train model
def train_tr_model(X, y, verbose = 0):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=7)

    model = svm.LinearSVC()
    start = time.time()
    model.fit(X_train, y_train)
    end = time.time()
    print("Single SVC, training time: {t}s, score: {s}".format(t = end - start, s = model.score(X_test, y_test)))
    return model

if __name__ == "__main__":

    text_windows = os.path.join(config.WINDOW_PATH, 'true/')
    ntext_windows = os.path.join(config.WINDOW_PATH, 'false/')
    
    X, y = prepare_tr_training_data(text_windows, ntext_windows)
    tr_model = train_tr_model(X, y)
    save_tr_model(tr_model, config.TEXT_MODEL_PATH)
