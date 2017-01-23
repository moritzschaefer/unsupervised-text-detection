#!/bin/bash
set -e
echo "This script downloads/unpacks data and runs everything"

mkdir -p data
cd data

mkdir -p patches
mkdir -p windows/true
mkdir -p windows/false

echo "download necessary data files"

curl "http://www.iapr-tc11.org/dataset/ICDAR2003_RobustReading/TrialTrain/word.zip" -o "word.zip"
unzip word.zip
rm word.zip
clear

curl "http://www.iapr-tc11.org/dataset/ICDAR2003_RobustReading/TrialTrain/scene.zip" -o "scene.zip"
unzip scene.zip
rm scene.zip
clear


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
python3 main.py
