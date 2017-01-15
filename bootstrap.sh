#!/bin/bash
set -e
echo "This script downloads/unpacks data and runs everything"

mkdir -p data
cd data

mkdir -p patches
mkdir -p windows

echo "download necessary data files"

curl "http://www.iapr-tc11.org/dataset/ICDAR2003_RobustReading/TrialTrain/word.zip" -o "word.zip"
unzip word.zip
rm word.zip
clear

curl "http://www.iapr-tc11.org/dataset/ICDAR2003_RobustReading/TrialTrain/scene.zip" -o "scene.zip"
unzip scene.zip
rm scene.zip
clear

cd ../src

echo "extracting random word patches for training"
python3 randomPatch_extraction.py

echo "learning feature dictionary"
python3 feature_learning.py

echo "expanding feature representation to every scenery image"
python3 feature_extraction.py

echo "finished feature extraction"
