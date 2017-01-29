import os


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                    os.path.pardir))
XML_PATH = os.path.join(BASE_DIR, 'data/SceneTrialTrain/locations.xml')
TEXT_PATH = os.path.join(BASE_DIR, 'data/word/')
SCENERY_PATH = os.path.join(BASE_DIR, 'data/SceneTrialTrain/')
PATCH_PATH = os.path.join(BASE_DIR, 'data/patches/')
WINDOW_PATH = os.path.join(BASE_DIR, 'data/windows/')
DICT_PATH = os.path.join(BASE_DIR, 'data/dict.npy')
NUM_PATCHES_PER_TEXT = 50  # TODO this should be ~100, for classification training purposes 30 atm
TOTAL_WINDOWS_FOR_TRAINING = 20000
ALPHA = .5 #hyperparam from feature extraction #TODO tune this in crossval

NUM_D = 200  # number of dictionary entries
