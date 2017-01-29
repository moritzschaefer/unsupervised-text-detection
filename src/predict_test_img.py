import cv2
import feature_extraction
import numpy as np
import config
import os
import glob
from skimage.transform import pyramid_gaussian
import pickle


# TODO use a path from config
dictionary = np.load(config.DICT_PATH)  # get dictionary


# get all windows of a image
def sliding_window(img, step_size):
    # windows=[]
    for x in range(0, img.shape[0], step_size):
        for y in range(0, img.shape[1], step_size):
            yield img[x:x + 32, y:y + 32]
            # windows.append(img[x:x+32,y:y+32])
    # return windows


# return the value of every pixels of a image
def get_prediction_values(img, model):
    values = []
    for win in sliding_window(img, 1):
        features = feature_extraction.get_features_for_window(dictionary, win)
        v = model.predict(features)
        values.append(v)
    values = np.array(values).reshape(img.shape)
    return values


# return all Scaling image of a Image,save into Layer Matrix
def get_all_layers(img):
    layer = []
    for (i, resized) in enumerate(pyramid_gaussian(img, downscale=2)):
        # if the image is too small, break from the loop
        if resized.shape[0] < 32 or resized.shape[1] < 32:
            break
        layer.append(resized)
    return layer

if __name__ == "__main__":
    model = pickle.load(open(config.TEXT_MODEL_PATH), 'rb')  # get model
    image_files = glob.glob(os.path.join(config.TEST_IMAGE_PATH, '/*.jpg'))

    for filename in image_files:
        img = cv2.imread(filename)
        # layer = get_all_layers(f)  # get all scaling of every image
        # value_all_layer = []
        predictions = get_prediction_values(img, model)

        # now draw the image
        text_probability_image = np.zeros(img.shape, np.uint8)

        for y in range(0, img.shape[0]):
            for x in range(0, img.shape[1]):
                window = predictions[max(0, y-32):min(y, predictions.shape[0]),
                                     max(0, x-32):min(x, predictions.shape[1])]
                text_probability_image[y, x] = max(window)

                cv2.imshow("image 1", img)
                cv2.imshow("image 2", text_probability_image)
                cv2.waitKey(0)

        # TODO: layer
        # for l in layer:  # for every layer
        #     # get the value of each pixels in every layer
        #     value = get_predict_value(f, model)
        #     value_all_layer.append()  # save value of every layer into value_AllLayer
