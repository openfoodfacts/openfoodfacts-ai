#!/bin/bash

set -euo pipefail

INPUT_DIR=$1
OUTPUT_DIR=$2

mkdir -p $OUTPUT_DIR
optimum-cli export onnx -m $INPUT_DIR --task token-classification --monolith --framework pt --opset 17 "${OUTPUT_DIR}/model.onnx"