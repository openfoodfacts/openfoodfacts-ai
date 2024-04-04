import re

from annotated_text import annotated_text
import requests
import streamlit as st


BARCODE_PATH_REGEX = re.compile(r"^(...)(...)(...)(.*)$")


def split_barcode(barcode: str) -> list[str]:
    """Split barcode in the same way as done by Product Opener to generate a
    product image folder.

    :param barcode: The barcode of the product. For the pro platform only,
        it must be prefixed with the org ID using the format
        `{ORG_ID}/{BARCODE}`
    :raises ValueError: raise a ValueError if `barcode` is invalid
    :return: a list containing the splitted barcode
    """
    org_id = None
    if "/" in barcode:
        # For the pro platform, `barcode` is expected to be in the format
        # `{ORG_ID}/{BARCODE}` (ex: `org-lea-nature/3307130803004`)
        org_id, barcode = barcode.split("/", maxsplit=1)

    if not barcode.isdigit():
        raise ValueError(f"unknown barcode format: {barcode}")

    match = BARCODE_PATH_REGEX.fullmatch(barcode)

    splits = [x for x in match.groups() if x] if match else [barcode]

    if org_id is not None:
        # For the pro platform only, images and OCRs belonging to an org
        # are stored in a folder named after the org for all its products, ex:
        # https://images.pro.openfoodfacts.org/images/products/org-lea-nature/330/713/080/3004/1.jpg
        splits.append(org_id)

    return splits


def _generate_file_path(barcode: str, image_id: str, suffix: str):
    splitted_barcode = split_barcode(barcode)
    return f"/{'/'.join(splitted_barcode)}/{image_id}{suffix}"


def generate_ocr_path(barcode: str, image_id: str) -> str:
    return _generate_file_path(barcode, image_id, ".json")


def generate_image_path(barcode: str, image_id: str) -> str:
    return _generate_file_path(barcode, image_id, ".400.jpg")


@st.cache_data
def send_prediction_request(ocr_url: str):
    return requests.get(
        "https://robotoff.openfoodfacts.net/api/v1/predict/ingredient_list",
        params={"ocr_url": ocr_url},
    ).json()


def get_product(barcode: str):
    r = requests.get(f"https://world.openfoodfacts.org/api/v2/product/{barcode}")

    if r.status_code == 404:
        return None

    return r.json()["product"]


def display_ner_tags(text: str, entities: list[dict]):
    spans = []
    previous_idx = 0
    for entity in entities:
        score = entity["score"]
        start_idx = entity["start"]
        end_idx = entity["end"]
        spans.append(text[previous_idx:start_idx])
        spans.append((text[start_idx:end_idx], f"{score:.3f}"))
        previous_idx = end_idx
    spans.append(text[previous_idx:])
    annotated_text(spans)


def run(barcode: str, min_threshold: float = 0.5):
    product = get_product(barcode)

    if not product:
        st.error(f"Product {barcode} not found")
        return

    images = product["images"]
    for image_id, _ in images.items():
        if not image_id.isdigit():
            continue

        ocr_path = generate_ocr_path(barcode, image_id)
        ocr_url = f"https://static.openfoodfacts.org/images/products{ocr_path}"
        prediction = send_prediction_request(ocr_url)

        entities = prediction["entities"]
        text = prediction["text"]
        filtered_entities = [e for e in entities if e["score"] >= min_threshold]

        if filtered_entities:
            st.divider()
            image_path = generate_image_path(barcode, image_id)
            image_url = f"https://static.openfoodfacts.org/images/products{image_path}"
            st.image(image_url)
            display_ner_tags(text, filtered_entities)


query_params = st.experimental_get_query_params()
default_barcode = query_params["barcode"][0] if "barcode" in query_params else ""

barcode = st.text_input("barcode", help="Barcode of the product", value=default_barcode)
threshold = st.number_input(
    "threshold",
    help="Minimum threshold for entity predictions",
    min_value=0.0,
    max_value=1.0,
    value=0.98,
)

if barcode:
    run(barcode, threshold)
