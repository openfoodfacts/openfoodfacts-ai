python scripts/dags/extract_from_argilla.py run \
    --deploy_to_hf true \
    --local_path data/dataset/deployed_data.parquet \
    --argilla_dataset_name training_dataset \
    --dataset_hf_repo openfoodfacts/spellcheck-dataset \
    --dataset_revision v4 \
    --dataset_test_size 0.1 \
    --dataset_version v4.3 \
    --status submitted