import cv2
import config
import random
import glob
from uuid import uuid4
import os

def extract_random_patches(numPatches_perimg):
    '''
    Return random patches for all images
    '''
    image_files = glob.glob(os.path.join(config.TEXT_PATH, '*.JPG'))

    for f in image_files:

        img = cv2.imread(f)

        for i in range(numPatches_perimg):
            x=random.randint(0, img.shape[0])
            y=random.randint(0, img.shape[1])
            patch=img[x:x+8,y:y+8]
            cv2.imwrite('{}/{}.JPG'.format(config.PATCH_PATH, uuid4()),patch)
        break
    print('finished window extraction.')
if __name__=="__main__":
    
    extract_random_patches(100)