"""
this file extracts windows w or w/o text for training purposes
"""
import config
import time
import os
import glob
import numpy as np
import pickle
import xml.etree.ElementTree
import logging
from pathlib import Path
import cv2
import random

from uuid import uuid4

logging.basicConfig(level=logging.INFO)


def count_images(path):
    """
    counts images
    """
    c = 0

    image_folders = glob.glob(os.path.join(path, '*/'))

    for folder in image_folders:
        image_files = glob.glob(os.path.join(folder, '*.JPG'))
        image_files.append(glob.glob(os.path.join(folder, '*.jpg')))
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
    WINDOWS ARE NOT PREPROCESSED!
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
    #img = preprocessing.preprocess(img)

    # check for img reading errors
    if img is None:
        return

    if plot:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
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
            window = (x, y, img[y:y + 32, x:x + 32])

            if window[2].shape != (32, 32, 3):
                print('-----------------------')
                print('text_window malformed: {}'.format(window[2].shape))
                print('image size: {}'.format(img.shape))
                print('sampled y: {}, sampled x: {}'.format(window[1], window[0]))

            if plot:
                rect = patches.Rectangle((window[0], window[1]),32,32,linewidth=1,edgecolor='r',facecolor='none')
                ax.add_patch(rect)
            np.save('{}/true/{}.npy'.format(config.WINDOW_PATH, uuid4()), window[2])

    else:

        restricted_x_coordinates = []
        restricted_y_coordinates = []

        #get all textbox coordinates
        for box in text_boxes:
            restricted_x_coordinates.append(list(range(box[0] - 32, box[0]+ box[1])))
            restricted_y_coordinates.append(list(range(box[2] - 32, box[2]+ box[3])))

        possible_x = list(range(0, img.shape[1] - 32))
        possible_y = list(range(0, img.shape[0] - 32))

        # areas with text
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
            window = (x, y, img[y:y + 32, x:x + 32])

            if window[2].shape != (32, 32, 3):
                print('-----------------------')
                print('n_text_window malformed: {}'.format(window[2].shape))
                print('image size: {}'.format(img.shape))
                print('sampled y: {}, sampled x: {}'.format(window[1], window[0]))

            if plot:
                rect = patches.Rectangle((window[0], window[1]),32,32,linewidth=1,edgecolor='r',facecolor='none')
                ax.add_patch(rect)
            np.save('{}/false/{}.npy'.format(config.WINDOW_PATH, uuid4()), window[2])

        if plot:
            plt.show()


def create_windows_for_training():
    # get meta info xml
    e = xml.etree.ElementTree.parse(config.XML_PATH).getroot()
    dic = parse_xml(e)
    n_images = count_images(config.SCENERY_PATH)
    win_per_image = config.TOTAL_WINDOWS_FOR_TRAINING // n_images

    logging.info('extracting {} windows for training.'.format(config.TOTAL_WINDOWS_FOR_TRAINING))

    image_folders = glob.glob(os.path.join(config.SCENERY_PATH, '*/'))

    for folder in image_folders:

        image_files = glob.glob(os.path.join(folder, '*.JPG'))

        for f in image_files:
            extract_random_windows(f, 32, (32, 32), win_per_image // 2, dic, text = True, plot = False)
            extract_random_windows(f, 32, (32, 32), win_per_image // 2, dic, text = False, plot = False)


if __name__ == "__main__":
    create_windows_for_training()
