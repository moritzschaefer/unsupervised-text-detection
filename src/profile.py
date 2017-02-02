import cv2
import feature_extraction
import numpy as np
import config
import cProfile

D = np.load(config.DICT_PATH)

path = '../data/windows/true/0a1bfc40-c6a1-4de4-b89e-a4f8764e5983.npy'

if __name__ == "__main__":
    feature_extraction.get_features_for_window(D, path)
