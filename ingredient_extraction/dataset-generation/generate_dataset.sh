#!/bin/bash

set -euo pipefail
IFS=$'\n\t'

VERSION=$1
python3 generate_dataset.py
mkdir -p datasets

zcat dataset.jsonl.gz | jq -c 'select(.meta.in_test_split == true)' | gzip > datasets/ingredient_detection_dataset-${VERSION}_test.jsonl.gz
zcat dataset.jsonl.gz | jq -c 'select(.meta.in_test_split == false)' | gzip > datasets/ingredient_detection_dataset-${VERSION}_train.jsonl.gz

scp datasets/ingredient_detection_dataset-${VERSION}_t* off-prod:/srv/off/html/data/datasets/