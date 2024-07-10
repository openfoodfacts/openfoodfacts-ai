#!/bin/bash

# Bash init script for SSP Cloud (https://datalab.sspcloud.fr/), used
# to set up the environment for a user.

# We save all logs in log.out to debug
exec 3>&1 4>&2
trap 'exec 2>&4 1>&3' 0 1 2 3
exec 1>log.out 2>&1

# Same for envvar
env | sort > env_init.out

# And for the used init script
wget -O init_originel.sh ${PERSONAL_INIT_SCRIPT}

# We retrieve the SSP Cloud username from the vault dir
export SSP_USER_NAME=${VAULT_TOP_DIR:1}

# Go to the work directory
if [[ -d "work" ]]
then
  cd work
fi

# Install some useful packages
apt update && apt install -y tmux htop

# This is required for Ultralytics package
apt install -y ffmpeg libsm6 libxext6

# Clone OpenFoodFacts AI
git clone --depth=1 https://github.com/openfoodfacts/openfoodfacts-ai.git

export folder="/home/onyxia/work/openfoodfacts-ai"

sudo -u ${USERNAME} sed -i "s/cd \/home\/onyxia\/work/cd \/home\/onyxia\/work\/openfoodfacts-ai/" /home/onyxia/.bashrc

# Final env (for debugging)
env | sort > env_final.out
