"""extraction functions to handle images(patches / windows)
or feature representations"""

import cv2
import config
import random
import glob
from uuid import uuid4
import os
import xml.etree.ElementTree
import random
import operator
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import preprocessing

def add_feature_data(windowspath, X=[]):
    """
    adds feature representations from windowspath to dataset
    """
    for window in windowspath:
        w = np.load(window)
        X.append(np.array(w).flatten())
    return X

def extract_random_patches(path, patches, rescale = False):
    '''
    Return random patches for img in path
    @TODO: Sometimes returns patches smaller than 8x8
    '''
    img = cv2.imread(path)
    img = preprocessing.preprocess(img)

    if rescale:
        img = cv2.rescale(img, (img.shape[1], 32))

    for i in range(patches):
        x = random.randint(0, img.shape[0] - 8)
        y = random.randint(0, img.shape[1] - 8)
        patch = img[x:x+8, y:y+8]
        cv2.imwrite('{}/{}.JPG'.format(config.PATCH_PATH, uuid4()), patch)

def count_images():
    """
    counts images
    """
    c = 0

    image_folders = glob.glob(os.path.join(config.SCENERY_PATH, '*/'))

    for folder in image_folders:
        image_files = glob.glob(os.path.join(folder, '*.JPG'))
        c += len(image_files)
    return c


def parse_xml(tree):
    dic = {}
    for child in tree.iter(tag='image'):
        img = child.getchildren()
        file_path = img[0].text
        res = img[1].attrib
        text_locations = [x.attrib for x in img[2].getchildren()]
        dic[file_path] = [res, text_locations]
    return dic


def extract_random_windows(path, stepSize, windowSize, windows, xmlDic, text, plot):
    '''
    Return random windows for given image
    '''


    #find path folder/image
    meta_name = Path(path).parts[-2] + '/' + Path(path).parts[-1]

    #get meta information for image if given
    try:
        meta = xmlDic[meta_name]
    except KeyError:
        return

    text_boxes = []

    #get all text boxes
    for box in meta[1]:
        # x coordinates of box
        x = int(float(box['x']))
        # width coordinates of box
        width = int(float(box['width']))
        #y coordinates of box
        y = int(float(box['y']))
        # height coordinates of box
        height = int(float(box['height']))
        text_boxes.append((x, width, y, height))

    if len(text_boxes) == 0:
        return

    img = cv2.imread(path)
    img = preprocessing.preprocess(img)

    # check for img reading errors
    if img == None:
        return

    if plot:
        fig,ax = plt.subplots(1)
        ax.imshow(img)

    if text:
        n_boxes = len(text_boxes)
        win_per_box = int(windows/n_boxes)
        valid_window = False

        # check for textboxes with at least 32x32
        for box in text_boxes:
            if (box[1] > 32 and box[3] > 32):
                valid_window = True
                break
            else:
                pass

        if not valid_window:
            return

        for i in range(windows):

            box = random.randint(0, n_boxes - 1)
            text_box = text_boxes[box]

            # choose textbox with sufficient size
            while (text_box[1] < 32 or text_box[3] < 32):
                box = random.randint(0, n_boxes - 1)
                text_box = text_boxes[box]

            x = random.randint(text_box[0], text_box[0] + text_box[1] - 32)
            y = random.randint(text_box[2], text_box[2] + text_box[3] - 32)
            window = (x, y, img[x:x + windowSize[0], y:y + windowSize[1]])
            if plot:
                rect = patches.Rectangle((window[0], window[1]),32,32,linewidth=1,edgecolor='r',facecolor='none')
                ax.add_patch(rect)
            cv2.imwrite('{}/true/{}.JPG'.format(config.WINDOW_PATH, uuid4()), window[2])

    else:

        restricted_x_coordinates = []
        restricted_y_coordinates = []

        #get all textbox coordinates
        for box in text_boxes:
            restricted_x_coordinates.append(list(range(box[0] - 32, box[0]+ box[1])))
            restricted_y_coordinates.append(list(range(box[2] - 32, box[2]+ box[3])))

        possible_x = list(range(0, img.shape[0] - 32))
        possible_y = list(range(0, img.shape[1] - 32))

        restricted_x_coordinates = [item for sublist in restricted_x_coordinates for item in sublist]
        restricted_y_coordinates = [item for sublist in restricted_y_coordinates for item in sublist]

        # allowed regions are those without text
        allowed_x = list(set(possible_x) - set(restricted_x_coordinates))
        allowed_y = list(set(possible_y) - set(restricted_y_coordinates))

        # if image has text everywhere
        if (len(allowed_x) == 0 or len(allowed_y) == 0):
            return

        for i in range(windows):
            x = random.choice(allowed_x)
            y = random.choice(allowed_y)
            window = (x, y, img[x:x + windowSize[0], y:y + windowSize[1]])

            if plot:
                rect = patches.Rectangle((window[0], window[1]),32,32,linewidth=1,edgecolor='r',facecolor='none')
                ax.add_patch(rect)
            cv2.imwrite('{}/false/{}.JPG'.format(config.WINDOW_PATH, uuid4()), window[2])

        if plot:
            plt.show()
