from pathlib import Path

import gradio as gr
import torch
from openfoodfacts import API
from openfoodfacts.images import (
    download_image,
    extract_barcode_from_path,
    extract_source_from_url,
    generate_json_ocr_url,
)
from openfoodfacts.ocr import OCRResult
from PIL import ImageDraw, ImageFont
from transformers import AutoModelForTokenClassification, AutoProcessor

NER_TAGS = [
    "O",
    "B-CALCIUM_100G",
    "I-CALCIUM_100G",
    "B-ENERGY_KJ_100G",
    "I-ENERGY_KJ_100G",
    "B-SATURATED_FAT_SERVING",
    "I-SATURATED_FAT_SERVING",
    "B-SUGARS_SERVING",
    "I-SUGARS_SERVING",
    "B-TRANS_FAT_100G",
    "I-TRANS_FAT_100G",
    "B-CALCIUM_SERVING",
    "I-CALCIUM_SERVING",
    "B-VITAMIN_D_100G",
    "I-VITAMIN_D_100G",
    "B-CARBOHYDRATES_100G",
    "I-CARBOHYDRATES_100G",
    "B-SODIUM_100G",
    "I-SODIUM_100G",
    "B-FAT_SERVING",
    "I-FAT_SERVING",
    "B-IRON_100G",
    "I-IRON_100G",
    "B-POTASSIUM_SERVING",
    "I-POTASSIUM_SERVING",
    "B-IRON_SERVING",
    "I-IRON_SERVING",
    "B-ENERGY_KCAL_100G",
    "I-ENERGY_KCAL_100G",
    "B-FIBER_100G",
    "I-FIBER_100G",
    "B-PROTEINS_SERVING",
    "I-PROTEINS_SERVING",
    "B-SALT_SERVING",
    "I-SALT_SERVING",
    "B-SUGARS_100G",
    "I-SUGARS_100G",
    "B-POTASSIUM_100G",
    "I-POTASSIUM_100G",
    "B-TRANS_FAT_SERVING",
    "I-TRANS_FAT_SERVING",
    "B-PROTEINS_100G",
    "I-PROTEINS_100G",
    "B-SALT_100G",
    "I-SALT_100G",
    "B-CHOLESTEROL_SERVING",
    "I-CHOLESTEROL_SERVING",
    "B-ENERGY_KJ_SERVING",
    "I-ENERGY_KJ_SERVING",
    "B-CARBOHYDRATES_SERVING",
    "I-CARBOHYDRATES_SERVING",
    "B-FIBER_SERVING",
    "I-FIBER_SERVING",
    "B-VITAMIN_D_SERVING",
    "I-VITAMIN_D_SERVING",
    "B-CHOLESTEROL_100G",
    "I-CHOLESTEROL_100G",
    "B-ADDED_SUGARS_SERVING",
    "I-ADDED_SUGARS_SERVING",
    "B-ENERGY_KCAL_SERVING",
    "I-ENERGY_KCAL_SERVING",
    "B-SERVING_SIZE",
    "I-SERVING_SIZE",
    "B-ADDED_SUGARS_100G",
    "I-ADDED_SUGARS_100G",
    "B-SATURATED_FAT_100G",
    "I-SATURATED_FAT_100G",
    "B-FAT_100G",
    "I-FAT_100G",
    "B-SODIUM_SERVING",
    "I-SODIUM_SERVING",
]
LABEL2ID = {label: i for i, label in enumerate(NER_TAGS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}


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


CHECKPOINT_DIR = "/home/raphael/Projects/models/nutrition-detector/ds-v4/"


def preprocess_ocr(ocr_result: OCRResult, width: int, height: int):
    if not ocr_result.full_text_annotation:
        return None, None

    items = [
        [word.text, word.bounding_poly.vertices]
        for page in ocr_result.full_text_annotation.pages
        for block in page.blocks
        for paragraph in block.paragraphs
        for word in paragraph.words
    ]
    words = []
    bboxes = []
    for word, vertices in items:
        words.append(word)
        x_min = int(1000 * min(v[0] for v in vertices) / width)
        x_max = int(1000 * max(v[0] for v in vertices) / width)
        y_min = int(1000 * min(v[1] for v in vertices) / height)
        y_max = int(1000 * max(v[1] for v in vertices) / height)
        bboxes.append(
            [
                max(0, min(999, x_min)),
                max(0, min(999, y_min)),
                max(0, min(999, x_max)),
                max(0, min(999, y_max)),
            ]
        )
    return words, bboxes


def run(image_identifier: str):
    if image_identifier.startswith("http"):
        image_source = extract_source_from_url(image_identifier)
        barcode = extract_barcode_from_path(image_source)
        image_id = Path(image_source).stem
    else:
        barcode, image_id = image_identifier.split("_")

    api = API(user_agent="Gradio Demo - Nutrition Detector")
    processor = AutoProcessor.from_pretrained(CHECKPOINT_DIR)
    model = AutoModelForTokenClassification.from_pretrained(CHECKPOINT_DIR)

    original_image_id = None
    if not image_id.isdigit():
        image_key = image_id.split(".")[0]
        product_images = api.product.get(barcode, fields=["images"])
        original_image_id = product_images[image_key]["imgid"]

    image = download_image((barcode, image_id), error_raise=False)
    if image is None:
        gr.Error("Error while fetching image")
        return

    if original_image_id is None:
        ocr_result = OCRResult.from_url(
            generate_json_ocr_url(barcode, image_id), error_raise=False
        )
    else:
        ocr_result = OCRResult.from_url(
            generate_json_ocr_url(barcode, original_image_id), error_raise=False
        )

    if ocr_result is None:
        gr.Error("Error while fetching OCR result")
        return

    words, bboxes = preprocess_ocr(ocr_result, image.width, image.height)

    if words is None:
        gr.Error("OCR result file is invalid")
        return

    encoding = processor(
        image,
        words,
        return_offsets_mapping=True,
        boxes=bboxes,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt",
    )
    offset_mapping = encoding.pop("offset_mapping").squeeze().numpy()
    is_subword = offset_mapping[:, 0] != 0

    with torch.inference_mode():
        # Logits is a tensor of shape
        # (sequence_length, num_labels)
        logits = model(**encoding).logits[0]
        attention_mask = encoding["attention_mask"][0].type(torch.bool)
        logits = logits[attention_mask]

    predictions = logits.argmax(dim=-1).squeeze().tolist()
    token_boxes = encoding["bbox"].squeeze().tolist()

    true_predictions = [
        ID2LABEL[pred] for idx, pred in enumerate(predictions) if not is_subword[idx]
    ]
    true_boxes = [
        unnormalize_box(box, image.width, image.height)
        for idx, box in enumerate(token_boxes)
        if not is_subword[idx]
    ]
    image = image.copy()
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=30)
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

    return image


demo = gr.Interface(
    fn=run,
    inputs=["text"],
    outputs=["image"],
)

demo.launch()
