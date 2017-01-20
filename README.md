# unsupervised-text-detection
Implementation of the paper "Text Detection and Character Recognition in Scene Images with Unsupervised Feature Learning" from Andrew Ng et. al.

# modules

## bootstrap

The bootstrap.sh script gets you started by unpacking the training data and running the different stages of the project.

- not used but possible: http://www.robots.ox.ac.uk/~vgg/data/text/

## patch extractor

Extracts all patches used for training. File patch_extraction.py
Note: The extracted text patches should have a size such that a 32x32 image roughly corresponds to the size of one character

## unsupervised dictionary builder

File feature_learning contains the k-means algorithm which learns the dictionary wich is used for the feature extraction.

## text recognition
...

...
