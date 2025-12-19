from pathlib import Path
from typing import Any

import requests

from llm_evals.datasets import insert_sample

BASE_URL = "https://prices.openfoodfacts.org"


def add_sample(inputs: str):
    """Add a new dataset sample to the current dataset."""
    price_tag_id = inputs
    item = fetch_sample(price_tag_id)
    dataset_path = Path(__file__).parent / "dataset.yaml"
    insert_sample(item, dataset_path)


def fetch_sample(price_tag_id: str) -> dict[str, Any]:
    r = requests.get(f"{BASE_URL}/api/v1/price-tags/{price_tag_id}")
    if not r.ok:
        raise ValueError(f"Invalid HTTP response: HTTP {r.status_code}")

    price_tag = r.json()
    if not (
        price_tag["predictions"]
        and any(
            prediction["type"] == "PRICE_TAG_EXTRACTION"
            for prediction in price_tag["predictions"]
        )
    ):
        raise ValueError(
            "Cannot add the sample: the sample does comply with the required "
            "conditions (at least one prediction of type PRICE_TAG_EXTRACTION "
            "and associated with a price ID)"
        )

    if "price_id" in price_tag:
        price = requests.get(f"{BASE_URL}/api/v1/prices/{price_tag['price_id']}").json()
    else:
        price = {}

    proof = price_tag.pop("proof")
    location = proof.pop("location")
    price_tag_image_path = price_tag["image_path"]
    price_tag_image_url = f"{BASE_URL}/img/{price_tag_image_path}"

    country_code = (
        location["osm_address_country_code"].lower()
        if location.get("osm_address_country_code")
        else "unknown"
    )
    tags = [
        f"country:{country_code}",
        f"currency:{price['currency'].lower() if price.get('currency') else 'unknown'}",
        f"type:{price['type'].lower() if price.get('type') else 'unknown'}",
    ]

    category = price.get("category_tag")
    if category:
        category = category.replace("en:", "").replace("-", " ").capitalize()
    return {
        "name": price_tag_id,
        "inputs": {"image_urls": [price_tag_image_url]},
        "metadata": {"tags": tags, "price_tag_id": int(price_tag_id)},
        "expected_output": {
            "type": price.get("type") or "PRODUCT",
            "product_code": price.get("product_code"),
            "category": [category],
            "labels_tags": price.get("labels_tags"),
            "origins_tags": price.get("origins_tags"),
            "price": price.get("price"),
            "price_is_discounted": price.get("price_is_discounted"),
            "price_without_discount": price.get("price_without_discount"),
            "discount_type": price.get("discount_type"),
            "price_per": price.get("price_per"),
            "currency": price.get("currency"),
        },
        "evaluators": [],
    }
