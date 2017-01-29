import cv2
import feature_extraction
import numpy as np
import config
import os
import glob
from skimage.transform import pyramid_gaussian
import pickle
from sklearn import svm


model=pickle.load(open("tr_model.sav"),'rb')#get model
dictionary=np.load(dict.npy)#get dictionary

def sliding_window(img,stepSize):#get all windows of a image
    windows=[]
    for x in range(0,img.shape[0],stepSize):
        for y in range(0,img.shape[1],stepSize):
            windows.append(img[x:x+32,y:y+32])
    return windows
        

def get_predictValue(img,model):#return the value of every pixels of a image
    cv2.load(img)
    windows=sliding_window(img,1)
    values=[]
    for win in windows:
        features=feature_extraction.get_features_for_window(dictionary, win)
        v=model.predict(features)
        values.append(v)
    values=values.reshape(img.shape)
    return values

def get_AllLayer(img):#return all Scaling image of a Image,save into Layer Matrix 
    Layer=[]
    for (i, resized) in enumerate(pyramid_gaussian(img, downscale=2)):
        # if the image is too small, break from the loop
        if resized.shape[0] < 32 or resized.shape[1] < 32:
            break
        Layer.append(resized)
    return Layer

if __name__=="__main__":
       
    image_folders = glob.glob(os.path.join(config.TESTIMAGE_PATH, '*/'))#here should indict exactly path where test image file store
    
    for folder in image_folders:
    
        image_files = glob.glob(os.path.join(folder, '*.jpg'))
    
        for f in image_files:
            Layer=get_AllLayer(f)#get all scaling of every image
            value_AllLayer=[]
            for l in Layer:#for every layer
                value=get_predictValue(f,model)#get the value of each pixels in every layer
                value_AllLayer.append()#save value of every layer into value_AllLayer

