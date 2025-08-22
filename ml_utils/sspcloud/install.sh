#!/bin/bash

set -euo pipefail
IFS=$'\n\t'

# Install some useful packages
apt update
apt install -y tmux htop

# This is required for Ultralytics package
apt install -y ffmpeg libsm6 libxext6
