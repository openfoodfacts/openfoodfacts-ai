"""Generate an image-text dataset compatible with LayoutLMv3 inputs from the
NER-like dataset."""

import logging
from io import BytesIO
from typing import Optional
from urllib.parse import urlparse

import PIL
import requests
from openfoodfacts import OCRResult
from openfoodfacts.ocr import Word
from openfoodfacts.utils import get_logger, http_session
from PIL import Image
from requests.exceptions import ConnectionError, SSLError, Timeout

import datasets
from datasets import DatasetDict, load_dataset

logger = get_logger()


class ImageLoadingException(Exception):
    """Exception raised by `get_image_from_url` when image cannot be fetched
    from URL or if loading failed.
    """

    pass


def _get_image_from_url(
    image_url: str,
    error_raise: bool = True,
    session: Optional[requests.Session] = None,
) -> Optional[requests.Response]:
    auth = (
        ("off", "off")
        if urlparse(image_url).netloc.endswith("openfoodfacts.net")
        else None
    )
    try:
        if session:
            r = session.get(image_url, auth=auth)
        else:
            r = requests.get(image_url, auth=auth)
    except (ConnectionError, SSLError, Timeout) as e:
        error_message = "Cannot download image %s"
        if error_raise:
            raise ImageLoadingException(error_message % image_url) from e
        logger.info(error_message, image_url, exc_info=e)
        return None

    if not r.ok:
        error_message = "Cannot download image %s: HTTP %s"
        error_args = (image_url, r.status_code)
        if error_raise:
            raise ImageLoadingException(error_message % error_args)
        logger.log(
            logging.INFO if r.status_code < 500 else logging.WARNING,
            error_message,
            *error_args,
        )
        return None

    return r


def get_image_from_url(image_url: str, error_raise: bool = True):
    s3_url = image_url.replace(
        "https://static.openfoodfacts.org/images/products/",
        "https://openfoodfacts-images.s3.eu-west-3.amazonaws.com/data/",
    )
    r = _get_image_from_url(s3_url, error_raise=error_raise)

    if r is None:
        logger.info(
            "Cannot download image from S3 (%s), falling back to Open Food Facts server",
            s3_url,
        )
        r = _get_image_from_url(image_url, error_raise=error_raise)

        if r is None:
            return None

    content_bytes = r.content
    try:
        return Image.open(BytesIO(content_bytes))
    except PIL.UnidentifiedImageError:
        error_message = f"Cannot identify image {image_url}"
        if error_raise:
            raise ImageLoadingException(error_message)
        logger.info(error_message)
    except PIL.Image.DecompressionBombError:
        error_message = f"Decompression bomb error for image {image_url}"
        if error_raise:
            raise ImageLoadingException(error_message)
        logger.info(error_message)


def generate_layoutlm_dataset_item(item):
    """Generate a LayoutLMv3 dataset item from a NER-like dataset item.

    :param item: the NER-like dataset item
    :return: the LayoutLMv3 dataset item
    """
    text = item["text"]
    offsets = item["offsets"]
    ocr_url = item["meta"]["url"]
    image_url = ocr_url.replace(".json", ".jpg")
    image = get_image_from_url(image_url, error_raise=False)
    new_item = {
        "text": text,
        "offsets": offsets,
        "meta": item["meta"],
        "words": [],
        "bboxes": [],
        "ner_tags": [],
    }

    if image is None:
        logger.info("Cannot load image from %s", image_url)
        return None

    if image.mode != "RGB":
        image = image.convert("RGB")

    ocr_result = OCRResult.from_url(ocr_url, http_session, error_raise=False)
    if ocr_result is None:
        logger.info("Cannot load OCR result from %s", ocr_url)
        return None

    first_words = set()
    selected_words = set()
    for i, (start_idx, end_idx) in enumerate(offsets):
        words: Optional[list[Word]] = ocr_result.get_words_from_indices(
            start_idx, end_idx
        )
        if words is None:
            logger.info(
                "Cannot get word indices #{%d} (%s) from OCR result %s",
                i,
                (start_idx, end_idx),
                ocr_url,
            )
            continue
        if len(words) == 0:
            raise ValueError("Empty word list")

        first_words.add(words[0])
        selected_words |= set(words)

    width, height = image.size
    for page in ocr_result.full_text_annotation.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                for word in paragraph.words:
                    if word in first_words:
                        ner_tag = "B-ING"
                    elif word in selected_words:
                        ner_tag = "I-ING"
                    else:
                        ner_tag = "O"
                    new_item["words"].append(word.text)
                    new_item["ner_tags"].append(ner_tag)
                    y_min = min([vertex[1] for vertex in word.bounding_poly.vertices])
                    x_min = min([vertex[0] for vertex in word.bounding_poly.vertices])
                    y_max = max([vertex[1] for vertex in word.bounding_poly.vertices])
                    x_max = max([vertex[0] for vertex in word.bounding_poly.vertices])
                    # Normalize bounding box coordinates: make sure that the
                    # coordinates don't overflow and are normalized between 0
                    # and 999
                    new_item["bboxes"].append(
                        (
                            max(0, min(999, int(x_min * 1000 / width))),
                            max(0, min(999, int(y_min * 1000 / height))),
                            max(0, min(999, int(x_max * 1000 / width))),
                            max(0, min(999, int(y_max * 1000 / height))),
                        )
                    )
                    new_item["image"] = image

    return new_item


DATASET_VERSION = "v6"

DATASET_URLS = {
    "train": f"https://static.openfoodfacts.org/data/datasets/ingredient_detection_dataset-alpha-{DATASET_VERSION}_train.jsonl.gz",
    "test": f"https://static.openfoodfacts.org/data/datasets/ingredient_detection_dataset-alpha-{DATASET_VERSION}_test.jsonl.gz",
}

base_ds = load_dataset("json", data_files=DATASET_URLS)

# Useful for debugging (checking if the image with bounding boxes is correct)
# for split_name in ("train", "test"):
#     for item in tqdm.tqdm(base_ds[split_name], desc="dataset items"):
#         new_item = generate_layoutlm_dataset_item(item)

#         if new_item["offsets"]:
#             display_image_bounding_boxes(
#                 new_item["image"], new_item["bboxes"], new_item["ner_tags"]
#             )

features = datasets.Features(
    {
        "ner_tags": datasets.Sequence(
            datasets.features.ClassLabel(names=["O", "B-ING", "I-ING"])
        ),
        "words": datasets.Sequence(datasets.Value("string")),
        "bboxes": datasets.Sequence(datasets.Sequence(datasets.Value("int64"))),
        "image": datasets.features.Image(),
        "text": datasets.features.Value("string"),
        "offsets": datasets.Sequence(datasets.Sequence(datasets.Value("int64"))),
        "meta": {
            "barcode": datasets.Value("string"),
            "image_id": datasets.Value("string"),
            "url": datasets.Value("string"),
            "id": datasets.Value("string"),
            "in_test_split": datasets.Value("bool"),
        },
    }
)
new_ds_train = base_ds["train"].map(
    generate_layoutlm_dataset_item,
    features=features,
    remove_columns=["marked_text", "tokens"],
)
new_ds_test = base_ds["test"].map(
    generate_layoutlm_dataset_item,
    features=features,
    remove_columns=["marked_text", "tokens"],
)

new_ds = DatasetDict({"train": new_ds_train, "test": new_ds_test})
new_ds.save_to_disk("datasets/ingredient-detection-layout-dataset-v1")
new_ds.push_to_hub("raphael0202/ingredient-detection-layout-dataset")
