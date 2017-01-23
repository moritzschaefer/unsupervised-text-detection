#!/bin/bash
set -e
echo "This script downloads/unpacks data and runs everything"

mkdir -p data
cd data

mkdir -p patches
mkdir -p windows

cd windows

mkdir -p true
mkdir -p false

cd ..

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

echo "run training module"
python3 main.py
