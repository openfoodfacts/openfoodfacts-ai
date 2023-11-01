#!/bin/bash

# max_steps=1600 > 20 epochs: 20 epochs * (5065 / 64)

WANDB_PROJECT=ingredient-extraction-layout WANDB_NAME=laymoutlmv3-large python train.py \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 32 \
  --model_name_or_path microsoft/layoutlmv3-large \
  --dataset_name ingredient-extraction \
  --output_dir layoutlmv3-large-ingredient-extraction \
  --do_train \
  --do_eval \
  --max_steps 1600 \
  --fp16 \
  --evaluation_strategy steps \
  --eval_steps 100 \
  --learning_rate 1e-5 \
  --load_best_model_at_end \
  --metric_for_best_model "eval_f1"