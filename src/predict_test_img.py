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
    for y in range(0, img.shape[0], step_size):
        for x in range(0, img.shape[1], step_size):
            yield img[y:min(y+32, img.shape[0]), x:min(x+32, img.shape[1])]
            # windows.append(img[x:x+32,y:y+32])
    # return windows


# return the value of every pixels of a image
def get_prediction_values(img, model):
    layers = []
    for layer_img in get_all_layers(img):
        values = []
        for window in sliding_window(layer_img, 1):
            features = feature_extraction.get_features_for_window(dictionary,
                                                                  window)
            v = model.predict(features)
            values.append(v)
        values = np.array(values).reshape(layer_img.shape)
        layers.append(values)
    return layers


# return all Scaling image of a Image,save into Layer Matrix
def get_all_layers(img):
    for (i, resized) in enumerate(pyramid_gaussian(img,
                                                   downscale=1.3,
                                                   max_layer=1)):  # TODO use 7
        # if the image is too small, break from the loop
        if resized.shape[0] < 32 or resized.shape[1] < 32:
            break
        yield resized

if __name__ == "__main__":
    model = pickle.load(open(config.TEXT_MODEL_PATH), 'rb')  # get model
    image_files = glob.glob(os.path.join(config.TEST_IMAGE_PATH, '/*.jpg'))

    for filename in image_files:
        img = cv2.imread(filename)
        prediction_layers = get_prediction_values(img, model)

        # now draw the image
        text_probability_image = np.zeros(img.shape, np.uint8)

        for y in range(0, img.shape[0]):
            for x in range(0, img.shape[1]):
                max_probability = 0
                for layer in prediction_layers:
                    # x and y in the layer which correspond to position in
                    # original image
                    trans_y = (layer.shape[0]/img.shape[0]) * y
                    trans_x = (layer.shape[1]/img.shape[1]) * y

                    window = layer[max(0, trans_y-32):
                                   min(trans_y, layer.shape[0]),
                                   max(0, trans_x-32):
                                   min(trans_x, layer.shape[1])]

                    max_probability = max(max_probability, max(window))

                text_probability_image[y, x] = max_probability

                cv2.imshow("image 1", img)
                cv2.imshow("image 2", text_probability_image)
                cv2.waitKey(0)
