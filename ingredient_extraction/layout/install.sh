#!/bin/bash

set -euo pipefail
IFS=$'\n\t'

# You can download and launch this script in one-line with:
# wget https://raw.githubusercontent.com/openfoodfacts/openfoodfacts-ai/develop/ingredient_extraction/layout/install.sh -O - | bash

# Install tmux and htop
sudo apt update
sudo apt install -y tmux htop

# Perform a blobless clone of the repository, it's the fastest way to download
# the training code (~2.5MB only)
git clone --filter=blob:none https://github.com/openfoodfacts/openfoodfacts-ai

cd openfoodfacts-ai/ingredient_extraction/layout

python3 -m pip install -r requirements.txt

echo "Install completed!"