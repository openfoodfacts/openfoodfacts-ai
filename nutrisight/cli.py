from pathlib import Path

import torch
import tqdm
import typer
from datasets import load_dataset
from PIL import ImageDraw, ImageFont
from transformers import AutoModelForTokenClassification, AutoProcessor

app = typer.Typer()


def unnormalize_box(bbox, width: int, height: int) -> tuple[float, float, float, float]:
    return (
        width * (bbox[0] / 1000),
        height * (bbox[1] / 1000),
        width * (bbox[2] / 1000),
        height * (bbox[3] / 1000),
    )


def iob_to_label(label: str) -> str:
    label = label[2:]
    if not label:
        return "other"
    return label


def label_to_color(label: str) -> str:
    if label == "other":
        return "white"

    if label.endswith("_100g"):
        return "green"

    return "purple"


@app.command()
def predict_from_dataset(model_name_or_path: str, output_dir: Path):
    processor = AutoProcessor.from_pretrained(model_name_or_path)
    model = AutoModelForTokenClassification.from_pretrained(model_name_or_path)

    ds = load_dataset("openfoodfacts/nutrient-detection-layout")
    val_ds = ds["test"]
    id2label = model.config.id2label

    def prepare_examples(examples):
        images = examples["image"]
        words = examples["tokens"]
        boxes = examples["bboxes"]
        encoding = processor(
            images,
            words,
            return_offsets_mapping=True,
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
        type="torch",
        columns=[
            "input_ids",
            "bbox",
            "attention_mask",
            "pixel_values",
            "offset_mapping",
        ],
    )

    output_dir.mkdir(exist_ok=True, parents=True)
    with torch.inference_mode():
        for i in tqdm.tqdm(range(len(prepared_val_ds)), desc="samples"):
            sample = val_ds[i]
            meta = sample["meta"]
            image = sample["image"]
            encoding = prepared_val_ds[i : i + 1]
            offset_mapping = encoding.pop("offset_mapping").squeeze().numpy()
            is_subword = offset_mapping[:, 0] != 0
            # Logits is a tensor of shape
            # (sequence_length, num_labels)
            logits = model(**encoding).logits[0]
            attention_mask = encoding["attention_mask"][0].type(torch.bool)
            logits = logits[attention_mask]
            predictions = logits.argmax(dim=-1).squeeze().tolist()
            token_boxes = encoding["bbox"].squeeze().tolist()

            true_predictions = [
                id2label[pred]
                for idx, pred in enumerate(predictions)
                if not is_subword[idx]
            ]
            true_boxes = [
                unnormalize_box(box, image.width, image.height)
                for idx, box in enumerate(token_boxes)
                if not is_subword[idx]
            ]
            image = image.copy()
            draw = ImageDraw.Draw(image)
            font = ImageFont.load_default()
            for prediction, box in zip(true_predictions, true_boxes):
                predicted_label = iob_to_label(prediction).lower()
                color = label_to_color(predicted_label)
                draw.rectangle(box, outline=color)
                draw.text(
                    (box[0] + 10, box[1] - 10),
                    text=predicted_label,
                    font=font,
                    fill=color,
                )

            image.save(output_dir / f"{meta['barcode']}_{meta['image_id']}.jpg")


if __name__ == "__main__":
    predict_from_dataset("openfoodfacts/nutrition-extractor", Path("ds-v5-large"))
