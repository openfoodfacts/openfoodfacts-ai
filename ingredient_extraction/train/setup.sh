#!/bin/bash

set -euo pipefail

# Install venv, uncomment if needed
# apt-get update -y && install python3.8-venv -y

# Check if the repository is already cloned
if [ -d "openfoodfacts-ai" ]; then
    echo "openfoodfacts-ai already cloned"
else
    echo "Cloning openfoodfacts-ai"
    # Perform a shallow git clone
    git clone https://github.com/openfoodfacts/openfoodfacts-ai.git --depth 1
fi

# Change directory to openfoodfacts-ai and 
cd openfoodfacts-ai/ingredient_extraction/train

# Create a virtual environment, uncomment if needed
# python3 -m venv .venv
# source .venv/bin/activate

# Install requirements
python3 -m pip install -r requirements.txt