# /// script
# dependencies = [
#   "typer",
#   "openfoodfacts>=2.9.0",
# ]
# ///
import time
from pathlib import Path
from random import choice

import typer

from openfoodfacts.api import API
from openfoodfacts.images import generate_image_url
from openfoodfacts.types import Facet


def main(
    countries: str,
    page_size: int = 200,
    limit_per_country: int = 25,
    output: Path = Path("images_urls.txt"),
):
    api = API(user_agent="ML dataset generation (raphael)", timeout=60)

    countries = [c.strip() for c in countries.strip().split(",")]
    seen_codes = set()

    with output.open("w") as f:
        for country in countries:
            print(f"Products for country: {country}")
            selected = 0
            products = api.facet.get_products(
                Facet.country,
                country,
                page=1,
                page_size=page_size,
                fields=["code", "images"],
            )["products"]
            for product in products:
                uploaded_images = product.get("images", {}).get("uploaded", [])
                if not uploaded_images:
                    continue
                barcode = product["code"]
                if barcode in seen_codes:
                    continue
                selected_image_id = choice(list(uploaded_images.keys()))
                image_url = generate_image_url(product["code"], selected_image_id)
                f.write(f"{image_url}\n")
                selected += 1
                seen_codes.add(barcode)
                if selected >= limit_per_country:
                    f.flush()
                    break
            # Be nice to the API
            time.sleep(6)


if __name__ == "__main__":
    typer.run(main)
