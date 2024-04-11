import gzip
from pathlib import Path
from datasets import load_dataset
import orjson
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
import wandb

ARTIFACT_NAME = (
    "raphaeloff/ingredient-detection-ner/model-xlm-roberta-large-20-epochs-alpha-v6:v0"
)

DATASET_URLS = {
    "train": "https://static.openfoodfacts.org/data/datasets/ingredient_detection_dataset-alpha-v6_train.jsonl.gz",
    "test": "https://static.openfoodfacts.org/data/datasets/ingredient_detection_dataset-alpha-v6_test.jsonl.gz",
}


api = wandb.Api()
artifact = api.artifact(ARTIFACT_NAME, type="model")
checkpoint_path = artifact.download()

tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

model = AutoModelForTokenClassification.from_pretrained(checkpoint_path)
dataset = load_dataset("json", data_files=DATASET_URLS)

classifier = pipeline(
    "ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple"
)

for split_name in ("test", "train"):
        split_ds = dataset[split_name]
        texts = split_ds["text"]
        aggregated_outputs = classifier(texts, batch_size=16)
        full_output = (
            {
                "text": split_ds[i]["text"],
                "meta": split_ds[i]["meta"],
                "entities": entities,
            }
            for i, entities in enumerate(aggregated_outputs)
        )
        prediction_file_path = Path(f"{split_name}_predictions_agg.jsonl.gz")
        with gzip.open(prediction_file_path, "wb") as f:
            f.write(
                b"\n".join(
                    orjson.dumps(item, option=orjson.OPT_SERIALIZE_NUMPY)
                    for item in full_output
                )
            )
