#!/bin/bash

# Bash init script for Google Cloud, used to set up the environment for a user.

# Add packages

apt update && apt install -y tmux htop

# Install minio

curl https://dl.min.io/client/mc/release/linux-amd64/mc \
  --create-dirs \
  -o /usr/local/bin/mc
chmod +x /usr/local/bin/mc

# Go to the work directory
USERNAME='raphael'
cd /home/${USERNAME}

# Clone OpenFoodFacts AI
sudo -u $USERNAME git clone --depth=1 https://github.com/openfoodfacts/openfoodfacts-ai.git

# Uncomment ll alias
sudo -u $USERNAME sed -i "s/#alias ll='ls -l'/alias ll='ls -l'/" /home/${USERNAME}/.bashrc

# Disable conda base environment auto-activation
sudo -u $USERNAME conda config --set auto_activate_base false