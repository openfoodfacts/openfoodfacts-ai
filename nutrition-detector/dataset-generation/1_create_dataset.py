import json
import random
import re
import string
from collections import defaultdict
from typing import Iterator, Optional

import openfoodfacts
import redis
import requests
import tqdm
import typer
from openfoodfacts.images import generate_image_url, generate_json_ocr_url
from openfoodfacts.ocr import OCRResult
from openfoodfacts.utils import get_logger
from ratelimit import limits, sleep_and_retry

logger = get_logger()

client = redis.Redis(host="localhost", port=6379, db=0)


def create_annotation_results(
    word_text: str, pre_annotation: str, vertices: list[tuple[int, int]], width, height
):
    x_min = min(v[0] for v in vertices) * 100
    x_max = max(v[0] for v in vertices) * 100
    y_min = min(v[1] for v in vertices) * 100
    y_max = max(v[1] for v in vertices) * 100

    id_ = "".join(random.choices(string.ascii_letters + string.digits, k=10))
    base_value = {
        "x": x_min / width,
        "y": y_min / height,
        "width": (x_max - x_min) / width,
        "height": (y_max - y_min) / height,
        "rotation": 0,
    }
    base_dict = {
        "id": id_,
        "to_name": "image",
        "image_rotation": 0,
        "original_width": width,
        "original_height": height,
    }
    return [
        {
            "type": "rectangle",
            "from_name": "bbox",
            "value": base_value,
            **base_dict,
        },
        {
            "type": "labels",
            "from_name": "label",
            "value": {"labels": [pre_annotation], **base_value},
            **base_dict,
        },
        {
            "type": "textarea",
            "from_name": "transcription",
            "value": {"text": [word_text], **base_value},
            **base_dict,
        },
    ]


NUTRIMENT_REGEX = re.compile(r"([0-9]+[,.]?[0-9]*) ?(kj|kcal|mg|g)")
NUTRIENT_NAMES = [
    "energy-kcal",
    "energy-kj",
    "proteins",
    "fat",
    "saturated-fat",
    "trans-fat",
    "carbohydrates",
    "sugars",
    "added-sugars",
    "fiber",
    "sodium",
    "salt",
    "cholesterol",
    "vitamin-d",
    "iron",
    "calcium",
    "potassium",
]


def generate_pre_annotation(word_texts: list[str], nutrients: dict):
    num_words = len(word_texts)
    mapping = defaultdict(set)
    for i in range(num_words):
        word = word_texts[i].strip().lower()
        if i < num_words - 1:
            next_word = word_texts[i + 1].strip().lower()
            concat = f"{word} {next_word}"
            # First try to match the word as is, then try to match the
            # concatenation of the word and the next word
            for candidate, indices_to_add in (
                (word, (i,)),
                (next_word, (i + 1,)),
                (concat, (i, i + 1)),
            ):
                if (match := NUTRIMENT_REGEX.search(candidate)) is not None:
                    value = match.group(1).replace(",", ".")
                    unit = match.group(2)
                    # Skip "100 g" as it's a common value
                    if value == "100" and unit == "g":
                        continue

                    mapping[(value, unit)].add(indices_to_add)
                    break

    logger.debug(f"mapping: {mapping.keys()}")

    annotated_words = []
    for nutrient in NUTRIENT_NAMES:
        for quantity in ("100g", "serving"):
            nutrient_key = f"{nutrient}_{quantity}"
            nutrient_quantity = nutrients.get(nutrient_key)
            if nutrient_quantity is None:
                continue

            nutrient_unit = nutrients[f"{nutrient}_unit"].lower()

            if int(nutrient_quantity) == nutrient_quantity:
                candidates = [
                    str(int(nutrient_quantity)),
                    f"{int(nutrient_quantity)}.0",
                ]
            else:
                candidates = [str(nutrient_quantity)]

            for nutrient_quantity_str in candidates:
                if (nutrient_quantity_str, nutrient_unit) in mapping:
                    logger.info(
                        f"Found {nutrient}_{quantity} in mapping ({nutrient_quantity_str} {nutrient_unit})"
                    )
                    if len(mapping[(nutrient_quantity_str, nutrient_unit)]) > 2:
                        logger.info(
                            f"Multiple annotations found for {nutrient}_{quantity}: {nutrient_quantity_str} {nutrient_unit}"
                        )
                        continue

                    word_indices = next(
                        iter(mapping[(nutrient_quantity_str, nutrient_unit)])
                    )
                    for word_idx in word_indices:
                        annotated_words.append((word_idx, nutrient_key))
                    break

    return annotated_words


def format_sample(product: dict, min_threshold: Optional[int] = None):
    barcode = product["code"]
    logger.info(f"Processing product {barcode}")
    main_lang = product["lang"]

    if "images" not in product:
        logger.info("Missing images")
        return None

    images = product["images"]
    nutrition_image_langs = list(
        set(
            [
                key.split("_")[1]
                for key in images.keys()
                if "nutrition" in key and "_" in key
            ]
        )
    )
    nutrition_image_langs = sorted(nutrition_image_langs, key=lambda x: x != main_lang)
    logger.info(
        f"Main lang: {main_lang}, Nutrition image langs: {nutrition_image_langs}"
    )

    for lang in nutrition_image_langs:
        if lang.startswith("new"):
            continue
        logger.info("Processing lang: %s", lang)
        nutrition_key = f"nutrition_{lang}"
        nutrition_image = images[nutrition_key]
        image_id = nutrition_image["imgid"]

        if image_id not in images:
            logger.info(f"Image {image_id} not found")
            continue

        full_image = images[image_id]
        width = full_image["sizes"]["full"]["w"]
        height = full_image["sizes"]["full"]["h"]
        image_url = generate_image_url(barcode, image_id=image_id)
        ocr_url = generate_json_ocr_url(barcode, image_id=image_id)

        try:
            ocr_result = OCRResult.from_url(ocr_url)
        except openfoodfacts.ocr.OCRResultGenerationException as e:
            logger.info(f"Error generating OCR result: {e}")
            continue

        if not ocr_result.full_text_annotation:
            continue

        words = [
            [word.text, word.bounding_poly.vertices, "other"]
            for page in ocr_result.full_text_annotation.pages
            for block in page.blocks
            for paragraph in block.paragraphs
            for word in paragraph.words
        ]

        pre_annotations = generate_pre_annotation(
            [w[0] for w in words], nutrients=product.get("nutriments", {})
        )
        matches = len(pre_annotations)

        for word_idx, nutrient in pre_annotations:
            words[word_idx][2] = nutrient

        annotation_results = []
        for word_text, vertices, pre_annotation in words:
            annotation_results += create_annotation_results(
                word_text, pre_annotation, vertices, width, height
            )

        if min_threshold is not None and matches < min_threshold:
            logger.info(
                f"Product {barcode} ('{nutrition_key}') has only {matches} matches, skipping image "
                f"(min threshold: {min_threshold})"
            )
            continue

        return {
            "data": {
                "image_url": image_url,
                "batch": "null",
                "meta": {"ocr_url": ocr_url},
            },
            "predictions": [{"result": annotation_results}],
        }

    return None


def product_iter(start_page: int, max_page: int) -> Iterator[str]:
    q = 'states_tags:"en:nutrition-photo-selected" AND states_tags:"en:nutrition-facts-completed"'
    for page in range(start_page, max_page + 1):
        r = requests.get(
            "https://search.openfoodfacts.org/search",
            params={"q": q, "page_size": 50, "page": page, "field": "code"},
        ).json()
        yield from (p["code"] for p in r["hits"])


@sleep_and_retry
@limits(calls=1, period=1)
def get_product(product_code: str):
    """Get product details from OpenFoodFacts API, and limit the rate of
    requests (1 req/s)."""
    return requests.get(
        f"https://world.openfoodfacts.org/api/v2/product/{product_code}?fields=images,nutriments,code,lang"
    ).json()


def create_dataset(
    start_page: int = 1,
    max_page: int = 100,
    min_threshold: int = 5,
    redis_cache_key: str = "off:nutrition-dataset:products",
):
    product_count = 0
    sample_count = 0

    with open("dataset.jsonl", "a") as f:
        for product_code in tqdm.tqdm(
            product_iter(start_page=start_page, max_page=max_page), desc="products"
        ):
            if client.hexists(redis_cache_key, product_code):
                logger.info(f"Product {product_code} already processed")
                continue
            product_count += 1
            r = get_product(product_code)

            if "product" not in r:
                logger.error(f"Product {product_code} not found")
                client.hset(redis_cache_key, product_code, "not_found")
                continue

            product = r["product"]
            sample = format_sample(product, min_threshold=min_threshold)
            if sample is not None:
                sample_count += 1
                client.hset(redis_cache_key, product_code, "added")
                f.write(json.dumps(sample) + "\n")
            else:
                client.hset(redis_cache_key, product_code, "discarded")

    logger.info(f"{product_count} products processed, {sample_count} samples generated")


if __name__ == "__main__":
    typer.run(create_dataset)
