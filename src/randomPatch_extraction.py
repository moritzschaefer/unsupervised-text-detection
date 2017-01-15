import cv2
import config
import random
import glob
from uuid import uuid4
import os

def extract_random_patches(numPatches_perImg):
    '''
    Return random patches for all images in TEXT_PATH
    '''
    image_folders = glob.glob(os.path.join(config.TEXT_PATH, '*/'))

    for folder in image_folders:

        image_files = glob.glob(os.path.join(folder, '*.jpg'))

        for f in image_files:
            img = cv2.imread(f)

            for i in range(numPatches_perImg):
                x=random.randint(0, img.shape[0])
                y=random.randint(0, img.shape[1])
                patch=img[x:x+8, y:y+8]
                cv2.imwrite('{}/{}.JPG'.format(config.PATCH_PATH, uuid4()), patch)
    print('finished random patch extraction.')

if __name__ == "__main__":
    try:
        os.mkdir(config.PATCH_PATH)
    except FileExistsError:
        pass
    extract_random_patches(config.NUM_PATCHES_PER_TEXT)
