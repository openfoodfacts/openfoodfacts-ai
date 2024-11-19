#!/bin/bash
RUN_NAME='ds-v5-large'
BASE_MODEL_NAME='microsoft/layoutlmv3-large'

DISABLE_MLFLOW_INTEGRATION="TRUE" WANDB_PROJECT=nutrition-detector WANDB_NAME=$RUN_NAME \
python3 \
train.py \
--output_dir $RUN_NAME \
--model_name_or_path $BASE_MODEL_NAME \
--do_train \
--do_eval \
--fp16 \
--per_device_train_batch_size 4 \
--gradient_accumulation_steps 8 \
--per_device_eval_batch_size 4 \
--eval_accumulation_steps 4 \
--load_best_model_at_end \
--save_total_limit 1 \
--max_steps 3000 \
--eval_steps 15 \
--logging_steps 15 \
--save_steps 15 \
--evaluation_strategy steps \
--save_strategy steps \
--metric_for_best_model "eval_f1" \
--learning_rate 1e-5 \
--push_to_hub \
--hub_model_id "openfoodfacts/nutrition-extractor" \
--hub_strategy "end"