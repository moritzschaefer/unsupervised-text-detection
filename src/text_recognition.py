"""
this codes trains, saves and loads a text recognition model
"""
import pickle
import glob
import os
import time
import logging

import numpy as np
import config
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
#test
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

logging.basicConfig(level=logging.INFO)

# load previously build model
def load_tr_model(path):
    loaded_model = pickle.load(open(path, 'rb'))
    return loaded_model


# save model
def save_tr_model(model, filename):
    pickle.dump(model, open(filename, 'wb'))


# prepare data
def prepare_tr_training_data(text_windows_path, n_text_windows_path):
    """

    """
    text_windows = glob.glob(os.path.join(text_windows_path, '*.npy'))
    ntext_windows = glob.glob(os.path.join(n_text_windows_path, '*.npy'))

    X = []

    n_text = 0
    for window in text_windows:
        w = np.load(window)
        # check for right dimensionality
        if w.shape == (3, 3, config.NUM_D):
            X.append(np.array(w).flatten())
            n_text += 1
        else:
            pass

    n_ntext = 0
    for window in ntext_windows:
        w = np.load(window)
        # check for right dimensionality
        if w.shape == (3, 3, config.NUM_D):
            X.append(np.array(w).flatten())
            n_ntext += 1
        else:
            pass

    y = [0] * (n_text + n_ntext)

    y[0:n_text] = [1] * n_text

    return shuffle(np.array(X), np.array(y))

# train model
def train_tr_model(X, y, verbose = 0):

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=7)

    # crossvalidation
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=7)

    # paramgrid
    param_grid=[{'C': [2**x for x in config.C_RANGE]}]

    # model
    model = svm.LinearSVC()
    #@TODO: Test other models:
    #model = LogisticRegression(solver='sag')
    # http://stackoverflow.com/questions/26478000/converting-linearsvcs-decision-function-to-probabilities-scikit-learn-python

    # gridsearch
    start = time.time()
    classifier = GridSearchCV(estimator=model, cv=cv, param_grid=param_grid, refit=True, verbose=3, n_jobs=1)
    classifier.fit(X_train, y_train)
    end = time.time()

    logging.info("GridSearch done, time: {t}s".format(t = end - start))

    best_C = classifier.best_params_['C']

    model = CalibratedClassifierCV(svm.LinearSVC(C=best_C))
    model.fit(X_train, y_train)

    logging.info("Prediction score: ", model.score(X_test, y_test))
    #print(model.predict_proba(X_test))

    return model

if __name__ == "__main__":

    text_windows = os.path.join(config.FEATURE_PATH, 'true/')
    ntext_windows = os.path.join(config.FEATURE_PATH, 'false/')

    X, y = prepare_tr_training_data(text_windows, ntext_windows)

    tr_model = train_tr_model(X, y)
    save_tr_model(tr_model, config.TEXT_MODEL_PATH)
