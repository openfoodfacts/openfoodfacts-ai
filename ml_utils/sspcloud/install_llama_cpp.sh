#!/bin/bash

set -euo pipefail
IFS=$'\n\t'

sudo apt update
sudo apt install -y cmake

git clone --depth=0 https://github.com/ggml-org/llama.cpp
cd llama.cpp

# Build llama.cpp
cmake -B build -DGGML_CUDA=ON -DBUILD_SHARED_LIBS=OFF
cmake --build build --config Release -j 16 --clean-first --target llama-cli llama-mtmd-cli llama-server llama-gguf-split
cp build/bin/llama-* ./
