import window_extraction
import feature_extraction
import text_recognition
import config
import cv2
import numpy as np
from sklearn import svm

# @TODO: implement argparse
"""
import optparse

parser = optparse.OptionParser()

parser.add_option('-q', '--query',
    action="store", dest="query",
    help="query string", default="spam")

options, args = parser.parse_args()

print 'Query string:', options.query
"""

# get all windows of a image

model = text_recognition.load_tr_model(config.TEXT_MODEL_PATH)

img_path = '../data/test_images/111-1152_IMG.jpg'

img = cv2.imread(img_path)

# prediction img
text_probability_image = np.zeros(img.shape[0:2], float)
print('shape of prob: ', text_probability_image.shape)

for y in range(0, img.shape[0]-32):
    for x in range(0, img.shape[1]-32):
        window = img[y:min(y+32, img.shape[0]), x:min(x+32, img.shape[1]), :]
        features = feature_extraction.get_features_for_window(window)
        if features[0]:
            features = features[1].flatten()
            # probability of text in window
            prediction = model.predict_proba(features.reshape(1, -1))[0][1]

            for (i, j), element in np.ndenumerate(text_probability_image[y:min(y+32, img.shape[0]), x:min(x+32, img.shape[1])]):
                if element < prediction:
                    text_probability_image[y + j][x + i] = prediction

            print(y, x)
        else:
            print('features malformed.')

np.save('../data/text_prediction_image.npy', text_probability_image)
