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

# Install cmake
sudo apt update
sudo apt install -y cmake

git clone --depth=1 https://github.com/ggml-org/llama.cpp
cd llama.cpp

# Build llama.cpp
cmake -B build -DGGML_CUDA=ON -DBUILD_SHARED_LIBS=OFF
cmake --build build --config Release -j 16 --clean-first --target llama-cli llama-mtmd-cli llama-server llama-gguf-split
cp build/bin/llama-* ./

# Launch llama-server
# ./llama-server \
#     -hf unsloth/Qwen3.5-9B-GGUF:UD-Q4_K_XL \
#     --ctx-size 16384 \
#     --temp 0.6 \
#     --top-p 0.95 \
#     --top-k 20 \
#     --min-p 0.00 \
#     --alias "unsloth/Qwen3.5-9B-GGUF" \
#     --port 8001 \
#     --reasoning off
