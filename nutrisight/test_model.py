from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForTokenClassification, AutoProcessor

checkpoint_dir = Path("/home/raphael/Projects/models/nutrition-detector/second-run/")
processor = AutoProcessor.from_pretrained(checkpoint_dir)
model = AutoModelForTokenClassification.from_pretrained(checkpoint_dir)

ds = load_dataset("openfoodfacts/nutrient-detection-layout")
val_ds = ds["test"]
all_ner_tags = val_ds.features["ner_tags"].feature.names
label2id = {label: i for i, label in enumerate(all_ner_tags)}
id2label = {i: label for label, i in label2id.items()}


def prepare_examples(examples):
    images = examples["image"]
    words = examples["tokens"]
    boxes = examples["bboxes"]
    encoding = processor(
        images,
        words,
        boxes=boxes,
        truncation=True,
        padding="max_length",
        max_length=512,
    )
    return encoding


prepared_val_ds = val_ds.map(
    prepare_examples, batched=True, remove_columns=val_ds.column_names
)
prepared_val_ds.set_format(
    type="torch", columns=["input_ids", "bbox", "attention_mask", "pixel_values"]
)

with torch.inference_mode():
    for i in range(len(prepared_val_ds)):
        encoding = prepared_val_ds[i : i + 1]
        # Logits is a tensor of shape
        # (sequence_length, num_labels)
        logits = model(**encoding).logits[0]
        attention_mask = encoding["attention_mask"][0]
        logits = logits[attention_mask]
        preds = logits.argmax(dim=-1)
        print(f"preds shape: {preds.shape}")
        break
