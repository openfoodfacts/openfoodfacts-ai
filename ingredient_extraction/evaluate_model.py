import gzip
import json
from pathlib import Path
from datasets import load_dataset
import wandb

from utils import generate_highlighted_text, fetch_annotations

ARTIFACT_NAME = "raphaeloff/ingredient-detection-ner/predictions:v12"
SPLIT_NAME = "test"

DATASET_URLS = {
    "train": "https://static.openfoodfacts.org/data/datasets/ingredient_detection_dataset-alpha-v6_train.jsonl.gz",
    "test": "https://static.openfoodfacts.org/data/datasets/ingredient_detection_dataset-alpha-v6_test.jsonl.gz",
}

base_ds = load_dataset("json", data_files=DATASET_URLS)

api = wandb.Api()
artifact = api.artifact(ARTIFACT_NAME, type="prediction")
prediction_dir = Path(artifact.download())


predictions = {}
with gzip.open(prediction_dir / f"{SPLIT_NAME}_predictions_agg_first.jsonl.gz", "rt") as f:
    for line in f:
        item = json.loads(line)
        id_ = item["meta"]["id"]
        predictions[id_] = item

gold = {}
for sample in base_ds[SPLIT_NAME]:
    gold[sample["meta"]["id"]] = dict(sample)


assert set(gold.keys()) == set(predictions.keys())

annotations = fetch_annotations()
correct = 0
incorrect = 0
manually_annotated_html_list = []
automatically_annotated_html_list = []

for key in gold.keys():
    gold_sample = gold[key]
    predict_sample = predictions[key]
    gold_offsets = gold_sample["offsets"]
    gold_markup = generate_highlighted_text(
        gold_sample["text"], gold_offsets, html_escape=True, mark_token="mark"
    )
    predicted_offsets = [[entity["start"], entity["end"]] for entity in predict_sample["entities"]]

    predict_markup = generate_highlighted_text(
        predict_sample["text"],
        predicted_offsets,
        html_escape=True,
        mark_token="mark",
    )

    if gold_markup == predict_markup:
        correct += 1
    else:
        incorrect += 1
        is_manually_annotated = key in annotations
        html_list = (
            manually_annotated_html_list
            if is_manually_annotated
            else automatically_annotated_html_list
        )
        predicted_offset_html = "</br>".join([f"\"{predict_sample['text'][start:end]}\" [{start}:{end}]" for start, end in predicted_offsets])
        html_list.append(
            f"<p>ID: {key}</br><b>Gold</b>:</br>{gold_markup}</br><b>Predicted</b>:</br>{predicted_offset_html}</br>{predict_markup}<p>"
        )

print(f"{correct=}, {incorrect=}")

html_list = [
    "<html><body><h1>Automatically annotated</h1>",
    *automatically_annotated_html_list,
    "<h1>Manually annotated</h1>",
    *manually_annotated_html_list,
    "</body></html>",
]

with open(f"compare_predictions_{SPLIT_NAME}.html", "wt") as f:
    f.write("\n".join(html_list))
