#!/bin/bash

WANDB_PROJECT=ingredient-extraction-layout WANDB_NAME=laymoutlmv3-base python train.py \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 16 \
  --gradient_accumulation_steps 8 \
  --model_name_or_path microsoft/layoutlmv3-base \
  --dataset_name ingredient-extraction \
  --output_dir layoutlmv3-ingredient-extraction \
  --do_train \
  --do_eval \
  # 20 epochs: 20 epochs * (5065 / 64)
  --max_steps 1600 \
  --fp16 \
  --evaluation_strategy steps \
  --eval_steps 100 \
  --learning_rate 1e-5 \
  --load_best_model_at_end \
  --metric_for_best_model "eval_f1"