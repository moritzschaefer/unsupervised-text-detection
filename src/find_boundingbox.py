from skimage.io import imread
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

def draw_boundingbox(img,thredhold):
    
    index=np.where(img>=thredhold)
    min_x=min(index[1])
    min_y=min(index[0])
    max_x=max(index[1])
    max_y=max(index[0])
    return min_x,min_y,max_x,max_y

img=imread("C:/Users/amiao/workspace/ML_PROJECT1/projectfiles/unsupervised-text-detection-feature_extraction/src/test_img.jpg")
thredhold=0.7
min_x,min_y,max_x,max_y=draw_boundingbox(img,thredhold)


width = max_x-min_x
height = max_y-min_y
# Create figure and axes
fig,ax = plt.subplots(1)
# Display the image
ax.imshow(img)
# Create a Rectangle patch
rect = patches.Rectangle((min_x,min_y),width,height,linewidth=1,fill=None,edgecolor='r')
# Add the patch to the Axes
ax.add_patch(rect)
plt.show()