#!/bin/bash

set -euo pipefail

apt-get update -y && install python3.8-venv -y

mkdir train && cd train

BASE_URL='https://raw.githubusercontent.com/openfoodfacts/openfoodfacts-ai/develop/ingredient_extraction/train'
wget ${BASE_URL}/train.py
wget ${BASE_URL}/requirements.txt

python3 -m venv .venv
source .venv/bin/activate

pip3 install -r requirements.txt