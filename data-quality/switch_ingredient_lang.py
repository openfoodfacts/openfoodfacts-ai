import csv
import enum
import logging
from pathlib import Path
from typing import Annotated, Optional

import backoff
import redis
import requests
import tqdm
import typer
from openfoodfacts import API
from openfoodfacts.api import parse_ingredients
from openfoodfacts.dataset import ProductDataset
from openfoodfacts.ingredients import add_ingredient_in_taxonomy_field
from openfoodfacts.taxonomy import Taxonomy, get_taxonomy
from openfoodfacts.types import Facet, JSONType, TaxonomyType
from openfoodfacts.utils import get_logger, http_session
from requests import HTTPError

logger = get_logger()
logger.addHandler(logging.FileHandler("switch_ingredient_lang.log"))


@backoff.on_exception(backoff.expo, requests.exceptions.RequestException, max_time=5)
def predict_lang(text: str, k: int = 10, threshold: float = 0.01) -> str:
    r = http_session.post(
        "https://robotoff.openfoodfacts.net/api/v1/predict/lang",
        data={"text": text, "k": k, "threshold": threshold},
    )

    r.raise_for_status()
    data = r.json()
    return data["predictions"]


def log_update(code: str, ingredients_text: str, original_lang: str, new_lang: str):
    output_path = Path("updates.csv")
    add_header = not output_path.exists()

    with open(output_path, "a", newline="") as csvfile:
        fieldnames = ["code", "ingredients_text", "original_lang", "new_lang"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if add_header:
            writer.writeheader()
        writer.writerow(
            {
                "code": code,
                "ingredients_text": ingredients_text,
                "original_lang": original_lang,
                "new_lang": new_lang,
            }
        )


@backoff.on_exception(backoff.expo, requests.exceptions.RequestException, max_time=5)
def update_ingredients(
    api: API,
    code: str,
    original_lang: str,
    new_lang: Optional[str],
    ingredients_text: str,
):
    body = {
        "code": code,
        f"ingredients_text_{original_lang}": "",
    }

    if new_lang is not None:
        body[f"ingredients_text_{new_lang}"] = ingredients_text
        body["comment"] = f"switching lang from '{original_lang}' to '{new_lang}'"
    else:
        body["comment"] = f"removing garbage ingredient list in '{original_lang}'"

    r = api.product.update(body)
    logger.info(f"Response: HTTP {r.status_code}")
    logger.info(r.text)


class LangFixProcessingStatus(str, enum.Enum):
    wrong_quality_tag = enum.auto()
    missing_ingredient_text = enum.auto()
    updated = enum.auto()
    unchanged = enum.auto()
    too_few_ingredients = enum.auto()


def process_product(
    api: API,
    product: JSONType,
    ingredient_taxonomy: Taxonomy,
    confirm: bool = False,
    ingredient_detection_threshold: float = 0.7,
    maximum_fraction_known_ingredients: Optional[float] = None,
    minimum_ingredients_n: int = 3,
) -> LangFixProcessingStatus:
    code = product["code"]
    lang = product["lang"]
    ingredients_lc = product.get("ingredients_lc", lang)
    ingredients_text = product.get(f"ingredients_text_{ingredients_lc}")

    if "ingredients" not in product:
        logger.info("No ingredients field")
        return LangFixProcessingStatus.unchanged

    parse_ingredients_original = product["ingredients"]
    (
        original_ingredients_n,
        original_known_ingredients_n,
    ) = add_ingredient_in_taxonomy_field(
        parse_ingredients_original, ingredient_taxonomy
    )

    if original_ingredients_n == 0:
        logger.info("No ingredients found")
        return LangFixProcessingStatus.unchanged

    original_fraction_known_ingredients = (
        original_known_ingredients_n / original_ingredients_n
    )

    if (
        maximum_fraction_known_ingredients is not None
        and original_fraction_known_ingredients > maximum_fraction_known_ingredients
    ):
        logger.info(
            "The ingredient list is already well recognized (%s > %s), skipping",
            original_fraction_known_ingredients,
            maximum_fraction_known_ingredients,
        )
        return LangFixProcessingStatus.wrong_quality_tag

    if original_ingredients_n < minimum_ingredients_n:
        return LangFixProcessingStatus.too_few_ingredients

    if not ingredients_text:
        logger.info(
            f"Missing ingredients_text_{ingredients_lc} for product {product['code']}"
        )
        return LangFixProcessingStatus.missing_ingredient_text

    logger.info(
        f"Product URL: https://world.openfoodfacts.org/product/{code}, lang: {lang}\n"
        f"ingredients_lc: {ingredients_lc}\ningredients_text: '''{ingredients_text}'''"
    )
    predicted_langs = [
        x for x in predict_lang(ingredients_text, threshold=0.05) if len(x["lang"]) == 2
    ]

    for predicted_lang in predicted_langs:
        predicted_lang_id = predicted_lang["lang"]
        predicted_confidence = predicted_lang["confidence"]

        if predicted_lang_id == ingredients_lc:
            logger.info(
                f"Predicted lang ('{predicted_lang_id}') is the same as ingredients_lc ('{ingredients_lc}')"
            )
            continue

        logger.info(f"Predicted lang: {predicted_lang_id} ({predicted_confidence})")
        parsed_ingredients = parse_ingredients(
            text=ingredients_text,
            lang=predicted_lang_id,
            api_config=api.api_config,
        )
        ingredients_n, known_ingredients_n = add_ingredient_in_taxonomy_field(
            parsed_ingredients, ingredient_taxonomy
        )

        if ingredients_n == 0:
            logger.info("No ingredients found")
            continue

        fraction_known_ingredients = known_ingredients_n / ingredients_n
        logger.info(
            f"% of recognized ingredients: {fraction_known_ingredients * 100}, "
            f"ingredients_n: {ingredients_n}, known_ingredients_n: {known_ingredients_n}"
        )

        if fraction_known_ingredients >= ingredient_detection_threshold:
            confirm_response = (
                typer.confirm(
                    f"Switching lang from '{ingredients_lc}' to '{predicted_lang_id}', confirm?"
                )
                if confirm
                else True
            )
            if confirm_response:
                update_ingredients(
                    api, code, ingredients_lc, predicted_lang_id, ingredients_text
                )
                log_update(code, ingredients_text, ingredients_lc, predicted_lang_id)
                return LangFixProcessingStatus.updated
            else:
                logger.info("Skipping")
                continue

    return LangFixProcessingStatus.unchanged


MAXIMUM_FRACTION_KNOWN_INGREDIENTS = {
    "en:ingredients-100-percent-unknown": 0.0,
    "en:ingredients-80-percent-unknown": 0.2,
    "en:ingredients-60-percent-unknown": 0.4,
    "en:ingredients-40-percent-unknown": 0.6,
    "en:ingredients-20-percent-unknown": 0.8,
}


def filter_dataset(
    dataset: ProductDataset, facet: Facet, facet_value: str, offset: int = 0
):
    field = f"{facet.value}_tags"

    for i, product in enumerate(tqdm.tqdm(dataset, desc="products")):
        if i < offset:
            continue
        if facet_value in product.get(field, []) and "code" in product:
            yield i, product["code"]


def main(
    username: Annotated[
        str, typer.Option(envvar="OFF_USERNAME", help="Username to use to login")
    ],
    password: Annotated[
        str, typer.Option(envvar="OFF_PASSWORD", help="Password to use to login")
    ],
    facet_name: str = typer.Option("data_quality_warning", help="Facet name"),
    facet_value: str = typer.Option(
        "en:ingredients-100-percent-unknown", help="Value of the facet"
    ),
    confirm: bool = typer.Option(False, help="Ask for a confirmation before updating"),
    timeout: int = typer.Option(30, help="Timeout for HTTP requests"),
    minimum_ingredients_n: int = typer.Option(
        3, help="Filter by minimum number of ingredients"
    ),
):
    redis_client = redis.Redis(host="localhost", port=6379)
    logger.info(f"Logged in as '{username}'")
    api = API(
        user_agent="Robotoff custom script",
        username=username,
        password=password,
        timeout=timeout,
    )
    ingredient_taxonomy = get_taxonomy(TaxonomyType.ingredient)
    redis_prefix = "ing-fix-100-percent-unknown"
    dataset = ProductDataset()
    facet_enum = Facet.from_str_or_enum(facet_name)
    offset = int(redis_client.get(f"{redis_prefix}:offset") or 0)
    barcode_iter = filter_dataset(dataset, facet_enum, facet_value, offset=offset)

    for offset, code in barcode_iter:
        redis_client.set(f"{redis_prefix}:offset", offset)
        try:
            product = api.product.get(code)
        except HTTPError as e:
            logger.warning(f"Skipping product {code}, error: {e}")

        if product is None:
            logger.info(f"Skipping product {code}, product not found")
            continue

        if redis_client.sismember(f"{redis_prefix}:codes", code):
            logger.info(f"Skipping already processed code: {code}")
            continue
        status = process_product(
            api,
            product,
            ingredient_taxonomy,
            confirm=confirm,
            maximum_fraction_known_ingredients=MAXIMUM_FRACTION_KNOWN_INGREDIENTS.get(
                facet_value
            ),
            minimum_ingredients_n=minimum_ingredients_n,
        )
        redis_client.sadd(f"{redis_prefix}:codes", code)
        redis_client.incr(f"{redis_prefix}:{status.name}")
        redis_client.incr(f"{redis_prefix}:total")


if __name__ == "__main__":
    typer.run(main)
