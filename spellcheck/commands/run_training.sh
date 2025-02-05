python scripts/dags/training/training.py run \
    --do_human_eval False \
    --evaluation_data_version v8.0 \
    --training_data_version v5.2 \
    --experiment_tag eval_loss --experiment_tag mistral-7b-v0.3 --experiment_tag eval-normalization --experiment_tag test
    