#!/bin/bash
set -e
echo "This script unpacks data and runs everything"

mkdir -p data
cd data
if [ ! -f msra.tar.bz2 ]; then
  echo "msra.tar.bz2 missing. Please download it (see WhatsApp) and move it to data/"
  exit 1
fi
tar -xjvf msra.tar.bz2
cd ../src
echo "Creating training patches"
python3 patch_extraction.py
echo "Done; Learning features"
python3 feature_learning.py
echo "Dictionary created. See config.py for directory information"
