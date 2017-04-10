import cv2
import numpy as np


def normalize(img):
    std = np.std(img)
    if std == 0:
        std = 1.0

    return (img-np.mean(img))/std


def zca(X):
    sigma = np.cov(X, rowvar=True)  # Correlation matrix
    # Singular Value Decomposition. X = U * np.diag(S) * V
    U, S, V = np.linalg.svd(sigma)
    epsilon = 0.1  # Whitening constant: prevents division by zero
    # ZCA Whitening matrix: U * Lambda * U'
    ZCAMatrix = np.dot(U, np.dot(np.diag(1.0/np.sqrt(S + epsilon)), U.T))
    return np.dot(ZCAMatrix, X)  # Data whitening


def preprocess(img):
    """
    Apply grayscale conversion, normalization and zca whitening
    """
    return zca(normalize(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)))
