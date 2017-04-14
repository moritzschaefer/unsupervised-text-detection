import os


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                        os.path.pardir))
DATA_DIR = os.path.join(BASE_DIR, 'data')
XML_PATH = os.path.join(BASE_DIR, 'data/SceneTrialTrain/locations.xml')
TEXT_PATH = os.path.join(BASE_DIR, 'data/word/')
SCENERY_PATH = os.path.join(BASE_DIR, 'data/SceneTrialTrain/')
PATCH_PATH = os.path.join(BASE_DIR, 'data/patches/')
WINDOW_PATH = os.path.join(BASE_DIR, 'data/windows/')
DICT_PATH = os.path.join(BASE_DIR, 'data/dict.npy')
FEATURE_PATH = os.path.join(BASE_DIR, 'data/features/')
CHARACTER_MODEL_PATH = os.path.join(BASE_DIR, 'data/character_model.pkl')
CONFUSION_MATRIX_PATH = os.path.join(BASE_DIR, 'data/confusion_matrix.npy')
NUM_PATCHES_PER_TEXT = 200
TEXT_MODEL_PATH = os.path.join(BASE_DIR, 'data/text_model.pkl')
TEST_IMAGE_PATH = os.path.join(BASE_DIR, 'data/test_images/test_set/')
TOTAL_WINDOWS_FOR_TRAINING = 60000  # min is 251
ALPHA = .5  # hyperparam from feature extraction #TODO tune this in crossval

NUM_D = 1000  # number of dictionary entries

LAYER_DOWNSCALE=1.5
NUM_LAYERS=0
# the thresholds depend on STEP_SIZE
STEP_SIZE = 4
TEXT_RECOGNITION_THRESHOLD = 36*4/STEP_SIZE

RESIZE_WORDS_FOR_DIC = True

C_RANGE = range(-5, 5) #range of regularization (error acceptance http://stats.stackexchange.com/questions/31066/what-is-the-influence-of-c-in-svms-with-linear-kernel)
