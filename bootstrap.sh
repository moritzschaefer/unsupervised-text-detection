#!/bin/bash
set -e
echo "This script downloads/unpacks data and runs everything"

mkdir -p data
cd data

mkdir -p patches
mkdir -p windows/true
mkdir -p windows/false
mkdir -p features
mkdir -p features/true
mkdir -p features/false

echo "download necessary data files"

if [ ! -f "word.zip" ]
then
  curl "http://www.iapr-tc11.org/dataset/ICDAR2003_RobustReading/TrialTrain/word.zip" -o "word.zip"
  unzip word.zip
fi

if [ ! -f "scene.zip" ]
then
  curl "http://www.iapr-tc11.org/dataset/ICDAR2003_RobustReading/TrialTrain/scene.zip" -o "scene.zip"
  unzip scene.zip
fi


# character recognition
if [ ! -f character_icdar_train.zip ]; then
  wget -O character_icdar_train.zip http://www.iapr-tc11.org/dataset/ICDAR2003_RobustReading/TrialTrain/char.zip
  wget -O character_icdar_test.zip http://www.iapr-tc11.org/dataset/ICDAR2003_RobustReading/TrialTest/char.zip
  mkdir -p character_icdar_train
  mkdir -p character_icdar_test
  unzip -d character_icdar_train character_icdar_train.zip
  unzip -d character_icdar_test character_icdar_test.zip
fi


cd ../src


echo "run training module"

echo "run patch extraction"
python3 random_patch_extraction.py

echo "run feature_learning"
python3 feature_learning.py

echo "run window extraction"
python3 window_extraction.py

echo "run feature extraction"
python3 feature_extraction.py

echo "run text_recognition"
python3 text_recognition.py

echo "python3 character_training.py"
