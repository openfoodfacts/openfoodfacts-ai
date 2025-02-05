import argilla as rg
from argilla._constants import DEFAULT_API_KEY

from datasets import load_dataset
from db import db
from utils import fetch_annotations

with db:
    annotations = fetch_annotations()

annotated_manually_ids = set(
    x["identifier"] for x in annotations.values() if x["action"] in ("a", "u")
)
rg.init(api_url="http://localhost:6900", api_key=DEFAULT_API_KEY)

dataset_version = "alpha-v6"
DATASET_URLS = {
    "train": f"https://static.openfoodfacts.org/data/datasets/ingredient_detection_dataset-{dataset_version}_train.jsonl.gz",
    "test": f"https://static.openfoodfacts.org/data/datasets/ingredient_detection_dataset-{dataset_version}_test.jsonl.gz",
}
base_ds = load_dataset("json", data_files=DATASET_URLS)
records = []

for split in ("train", "test"):
    ds = base_ds[split]
    for record in ds:
        id_ = record["meta"]["id"]
        metadata = {k: v for k, v in record["meta"].items() if k != "id"}
        metadata["split"] = split
        tokencat_record = rg.TokenClassificationRecord(
            text=record["text"],
            tokens=record["tokens"],
            prediction=[
                ("ING", start_idx, end_idx) for start_idx, end_idx in record["offsets"]
            ],
            prediction_agent="GPT-3.5",
            id=record["meta"]["id"],
            metadata=metadata,
            status="Validated" if id_ in annotated_manually_ids else "Default",
        )
        records.append(tokencat_record)

rg.log(records, "ingredient-detection-ner")
