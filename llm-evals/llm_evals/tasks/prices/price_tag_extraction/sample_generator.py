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
        and price_tag["price_id"] is not None
    ):
        raise ValueError(
            "Cannot add the sample: the sample does comply with the required "
            "conditions (at least one prediction of type PRICE_TAG_EXTRACTION, "
            "of schema version 2.0, and associated with a price ID)"
        )

    price = requests.get(f"{BASE_URL}/api/v1/prices/{price_tag['price_id']}").json()
    location = price.pop("location")
    price_tag["location"] = location
    price_tag["proof"].pop("location")
    price_tag_image_path = price_tag["image_path"]
    price_tag_image_url = f"{BASE_URL}/img/{price_tag_image_path}"

    country_code = (
        location["osm_address_country_code"].lower()
        if location.get("osm_address_country_code")
        else "unknown"
    )
    tags = [
        f"country:{country_code}",
        f"currency:{price['currency'].lower() if price['currency'] else 'unknown'}",
        f"type:{price['type'].lower()}",
    ]
    return {
        "name": price_tag_id,
        "inputs": {"image_urls": [price_tag_image_url]},
        "metadata": {"tags": tags, "price_tag_id": price_tag_id},
        "expected_output": {
            "type": price["type"],
            "product_code": price["product_code"],
            "category_tag": price["category_tag"],
            "labels_tags": price["labels_tags"],
            "origins_tags": price["origins_tags"],
            "price": price["price"],
            "price_is_discounted": price["price_is_discounted"],
            "price_without_discount": price["price_without_discount"],
            "discount_type": price["discount_type"],
            "price_per": price["price_per"],
            "currency": price["currency"],
        },
        "evaluators": [],
    }
