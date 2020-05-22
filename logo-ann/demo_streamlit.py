import json
import os
import pathlib
import random
import re
import tempfile
from typing import Any, Dict, List, Optional

import annoy
from PIL import Image
import requests
import streamlit as st


http_session = requests.Session()


LOCAL_DB = os.environ.get("LOCAL_DB", False)

if LOCAL_DB:
    ROBOTOFF_BASE_URL = "http://localhost:5500/api/v1"
else:
    ROBOTOFF_BASE_URL = "https://robotoff.openfoodfacts.org/api/v1"

PREDICTIONS_URL = ROBOTOFF_BASE_URL + "/images/predictions"
API_URL = "https://world.openfoodfacts.org/api/v0"
PRODUCT_URL = API_URL + "/product"
OFF_IMAGE_BASE_URL = "https://static.openfoodfacts.org/images/products"
BARCODE_PATH_REGEX = re.compile(r"^(...)(...)(...)(.*)$")


@st.cache(allow_output_mutation=True)
def load_index(file_path: pathlib.Path):
    index = annoy.AnnoyIndex(1280, "euclidean")
    index.load(str(file_path), prefault=True)
    return index


@st.cache(allow_output_mutation=True)
def load_keys(file_path: pathlib.Path):
    items = []
    with file_path.open("r") as f:
        for line in f:
            items.append(json.loads(line))

    return items


def get_image_from_url(
    image_url: str,
    error_raise: bool = False,
    session: Optional[requests.Session] = None,
) -> Optional[Image.Image]:
    if session:
        r = http_session.get(image_url)
    else:
        r = requests.get(image_url)

    if error_raise:
        r.raise_for_status()

    if r.status_code != 200:
        return None

    with tempfile.NamedTemporaryFile() as f:
        f.write(r.content)
        image = Image.open(f.name)

    return image


def split_barcode(barcode: str) -> List[str]:
    if not barcode.isdigit():
        raise ValueError("unknown barcode format: {}".format(barcode))

    match = BARCODE_PATH_REGEX.fullmatch(barcode)

    if match:
        return [x for x in match.groups() if x]

    return [barcode]


def generate_image_path(barcode: str, image_id: str) -> str:
    splitted_barcode = split_barcode(barcode)
    return "/{}/{}.jpg".format("/".join(splitted_barcode), image_id)


def display_predictions(
    index: annoy.AnnoyIndex,
    keys: List[Dict[str, Any]],
    count: int,
    min_confidence: Optional[float] = None,
    item_idx: Optional[int] = None,
):
    if not item_idx:
        item_idx = random.randint(0, len(keys) - 1)

    st.write("Item index: {}".format(item_idx))
    item = keys[item_idx]

    url = OFF_IMAGE_BASE_URL + generate_image_path(item["barcode"], item["image_id"])
    image = get_image_from_url(url, session=http_session)

    if image is None:
        return

    bounding_box = item["bounding_box"]
    ymin, xmin, ymax, xmax = bounding_box
    (left, right, top, bottom) = (
        xmin * image.width,
        xmax * image.width,
        ymin * image.height,
        ymax * image.height,
    )
    cropped_image = image.crop((left, top, right, bottom))
    st.image(cropped_image, width=400)

    closest_item_indexes, closest_item_distances = index.get_nns_by_item(
        item_idx, count, include_distances=True
    )
    cropped_images: List[Image.Image] = []
    captions: List[str] = []

    for closest_item_idx, distance in zip(closest_item_indexes, closest_item_distances):
        closest_item = keys[closest_item_idx]
        score = closest_item["confidence"]

        if min_confidence and score < min_confidence:
            continue

        url = OFF_IMAGE_BASE_URL + generate_image_path(
            closest_item["barcode"], closest_item["image_id"]
        )
        image = get_image_from_url(url, session=http_session)

        if image is None:
            continue

        bounding_box = closest_item["bounding_box"]
        ymin, xmin, ymax, xmax = bounding_box
        (left, right, top, bottom) = (
            xmin * image.width,
            xmax * image.width,
            ymin * image.height,
            ymax * image.height,
        )
        cropped_image = image.crop((left, top, right, bottom))

        if cropped_image.height > cropped_image.width:
            cropped_image = cropped_image.rotate(90)

        cropped_images.append(cropped_image)
        captions.append("score: {}, distance: {}".format(score, distance))

    if cropped_images:
        st.image(cropped_images, captions, width=200)


st.sidebar.title("Image Detection Demo")
count = st.sidebar.number_input(
    "Number of results to display", min_value=1, max_value=50, value=30
)
min_confidence = st.sidebar.number_input("Min confidence", format="%f") or None

index = load_index(pathlib.Path("data/index.bin"))
keys = load_keys(pathlib.Path("data/index.jsonl"))

item_idx = st.sidebar.number_input("Item index", min_value=1, step=1) or None

if item_idx is not None:
    item_idx -= 1

display_predictions(
    index=index,
    keys=keys,
    count=count,
    min_confidence=min_confidence,
    item_idx=item_idx,
)
