import numpy as np
import pandas as pd
import cv2
import glob
import os
import math
from uuid import uuid4
from preprocessing import preprocess
import config


def load_metadata():
    image_files = glob.glob(os.path.join(config.DATASET_PATH, '*.jpg'))
    dfs = []
    for f in image_files:
        filename = os.path.splitext(os.path.split(f)[1])[0]
        try:
            names = ['patch_number', 'difficult', 'x', 'y', 'w', 'h', 'angle']
            dtype = [np.int, np.bool, np.int, np.int, np.int, np.int, np.float]
            tmp_df = pd.read_csv(f[:-3] + 'gt',
                                 delimiter=' ',
                                 header=None,
                                 names=names,
                                 dtype={n: t for n, t in zip(names, dtype)})
        except:
            print('{} contains no text'.format(filename))
            continue
        tmp_df['filename'] = filename
        dfs.append(tmp_df)
    return pd.concat(dfs).reset_index()


def extract_random_patch(row, apply_preprocessing=True):
    '''
    Return a horizontal random 8 by 8 patch
    '''
    # load image
    img = cv2.imread(os.path.join(config.DATASET_PATH, row.filename + '.jpg'))
    rows, cols, dim = img.shape

    #TODO: check if rotated image is rotated the right way
    #cv2.imwrite('test.png', rotated_img)

    # rotate image to get horizontal
    y = row['y'] + (row['h'] / 2.0)
    x = row['x'] + (row['w'] / 2.0)
    M = cv2.getRotationMatrix2D((x, y), row.angle * 180.0 / math.pi, 1)
    rotated_img = cv2.warpAffine(img, M, (cols, rows))

    # cut text
    text_img = rotated_img[row.y:row.y+row.h, row.x:row.x+row.w]

    # extract random patches
    for _ in range(config.NUM_PATCHES_PER_TEXT):
        try:
            x, y = np.random.randint(0, row.w-8), np.random.randint(0, row.h-8)
        except ValueError:
            continue
        patch = text_img[y:y+8, x:x+8]
        if apply_preprocessing:
            patch = preprocess(patch)

        # save to file
        cv2.imwrite('{}/{}.png'.format(config.PATCH_PATH, uuid4()), patch)

if __name__ == "__main__":
    try:
        os.mkdir(config.PATCH_PATH)
    except FileExistsError:
        pass
    df = load_metadata()
    df.apply(extract_patch, axis=1)
