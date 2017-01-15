import os


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                    os.path.pardir))
TEXT_PATH = os.path.join(BASE_DIR, 'data/word/')
DATASET_PATH = os.path.join(BASE_DIR, 'data/SceneTrialTrain/lfsosa_12.08.2002/')
PATCH_PATH = os.path.join(BASE_DIR, 'data/patches/')
WINDOW_PATH = os.path.join(BASE_DIR, 'data/windows/')
DICT_PATH = os.path.join(BASE_DIR, 'data/dict.npy')
NUM_PATCHES_PER_TEXT = 10  # TODO this should be ~100
ALPHA = .5 #hyperparam from feature extraction

NUM_D = 20  # number of dictionary entries
