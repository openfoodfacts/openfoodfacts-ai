#!/bin/bash

set -euo pipefail
IFS=$'\n\t'

apt update
apt install -y tmux htop

# This is required for Ultralytics package
apt install -y ffmpeg libsm6 libxext6


git clone https://github.com/openfoodfacts/labelr.git

# Install uv

curl -LsSf https://astral.sh/uv/install.sh | sh

# Install direnv

curl -sfL https://direnv.net/install.sh | bash

## Post install

echo 'eval "$(direnv hook bash)"' >> ~/.bashrc