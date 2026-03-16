#!/bin/bash

set -euo pipefail
IFS=$'\n\t'

# Bash init script for SSP Cloud (https://datalab.sspcloud.fr/), used
# to set up the environment for a user.

# We save all logs in log.out for debugging
exec 3>&1 4>&2
trap 'exec 2>&4 1>&3' 0 1 2 3
exec 1>log.out 2>&1

# We retrieve the SSP Cloud username from the vault dir
export SSP_USER_NAME=${VAULT_TOP_DIR:1}

# Go to the work directory
if [[ -d "work" ]]
then
  cd work
fi

# Clone OpenFoodFacts AI
git clone https://github.com/openfoodfacts/labelr.git

# We change the default directory to labelr/packages/train-yolo, which is where the training code is located
sudo -u ${USERNAME} sed -i "s/cd \/home\/onyxia\/work/cd \/home\/onyxia\/work\/labelr\/packages\/train-yolo/" /home/onyxia/.bashrc

# Install uv

curl -LsSf https://astral.sh/uv/install.sh | sh

# Install direnv

curl -sfL https://direnv.net/install.sh | bash
echo 'eval "$(direnv hook bash)"' >> ~/.bashrc


# Add ~/.local/bin to PATH in .bashrc
if ! grep -q 'export PATH="$HOME/.local/bin:$PATH"' /home/onyxia/.bashrc; then
  echo 'export PATH="$HOME/.local/bin:$PATH"' >> /home/onyxia/.bashrc
fi

# Download install.sh script
wget -O install.sh https://raw.githubusercontent.com/openfoodfacts/openfoodfacts-ai/refs/heads/develop/ml_utils/sspcloud/install.sh
