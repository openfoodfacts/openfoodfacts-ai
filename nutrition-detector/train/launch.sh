DISABLE_MLFLOW_INTEGRATION="TRUE" WANDB_PROJECT=nutrition-detector WANDB_NAME='second-run' \
python3 \
train.py \
--output_dir second-run \
--model_name_or_path microsoft/layoutlmv3-base \
--do_train \
--do_eval \
--fp16 \
--per_device_train_batch_size 4 \
--gradient_accumulation_steps 8 \
--per_device_eval_batch_size 4 \
--eval_accumulation_steps 4 \
--load_best_model_at_end \
--save_total_limit 1 \
--max_steps 1600 \
--eval_steps 15 \
--logging_steps 15 \
--save_steps 15 \
--evaluation_strategy steps \
--save_strategy steps \
--metric_for_best_model "eval_f1" \
--learning_rate 1e-5
