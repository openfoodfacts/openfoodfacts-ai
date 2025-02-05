#!bin/bash

set -euo pipefail

# Create Vast.ai instance
# vast create instance 8683 \
# --image pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime \
# --ssh --onstart setup.sh --disk 73 \
# --direct

# 1 GPU
# python3 \
# train/train.py \
# xlm-roberta-large-20-epochs-alpha-v6-2 \
# --dataset-version alpha-v6 \
# --per-device-train-batch-size 8 \
# --gradient-accumulation-steps 8 \
# --per-device-eval-batch-size 32 \
# --num-train-epochs 20


# 2 GPUs
python3 \
train.py \
xlm-roberta-large-20-epochs-v1.1-alpha.1 \
--dataset-version v1.1-alpha.1 \
--per-device-train-batch-size 4 \
--gradient-accumulation-steps 8 \
--per-device-eval-batch-size 16 \
--num-train-epochs 20