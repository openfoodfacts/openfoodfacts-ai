import logging
import requests
from pathlib import Path
import os
from typing import List, Iterable, Mapping
import json
import random
from dataclasses import dataclass

from spellcheck import SpellChecker
from utils.llm import OpenAIChatCompletion
from utils.prompt import SystemPrompt, Prompt
from utils.model import OpenAIModel


LOGGER = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)

URL = "https://world.openfoodfacts.org/data-quality-warning/en:ingredients-100-percent-unknown.json?page={}"

SPELLCHECK_DIR = Path(os.path.dirname(__file__)).parent.parent
LABELED_DATA_PATH = SPELLCHECK_DIR / "data/labeled/corrected_list_of_ingredients.txt"
OLD_FR_DATA_PATH = SPELLCHECK_DIR / "data/fr/1_old_fr_no_duplicate_data.json"
BENCHMARK_PATH = SPELLCHECK_DIR / "data/benchmark/benchmark.json"

BENCHMARK_VERSION = "1.0.0"


def prepare_benchmark(
    benchmark_version: str,
    save_path: Path,
    spellchecker: SpellChecker,
    url: str = URL,
    n_pages: int = 5,
    feature_names: List[str] = ["ingredients_text", "lang"],
    labeled_data_path: Path = LABELED_DATA_PATH,
    old_fr_data_path: Path = OLD_FR_DATA_PATH,
    old_fr_data_sample_size: float = 0.3,
    random_seed: int = 42
) -> None:
    """"""
    data_from_labeled = prepare_data_from_labeled(
        labeled_data_path=labeled_data_path
    )
    data_from_old = prepare_data_from_old_fr(
        old_fr_data_path=old_fr_data_path,
        sample_size=old_fr_data_sample_size,
        random_seed=random_seed
    )
    data_from_db = prepare_data_from_db(
        url=url,
        n_pages=n_pages,
        spellchecker=spellchecker,
        feature_names=feature_names
    )
    data = {
        "version": benchmark_version,
        "data": data_from_labeled + data_from_old + data_from_db
    }
    # Saving
    with open(save_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def prepare_data_from_db(
        url: str,
        n_pages: int,
        spellchecker: SpellChecker,
        feature_names: List[str]
) -> Iterable[Mapping]:  
    """"""
    LOGGER.info("Start extracting and augmenting data from database.")
    if "ingredients_text" not in feature_names:
        raise ValueError(
            f"The feature 'ingredients_text' needs to be added to the feature names. Current features: {feature_names}"
        )
    products_data = []
    for page_number in range(n_pages):
        #TODO '<!DOCTYPE html><html><head><meta name="robots" content="noindex"></head><body><h1>NOINDEX</h1><p>
        # We detected that your browser is a web crawling bot, and this page should not be indexed by web crawlers. 
        # If this is unexpected, contact us on Slack or write us an email at 
        # <a href="mailto:contact@openfoodfacts.org">contact@openfoodfacts.org</a>.</p></body></html>'
        data = json.loads(requests.get(url=url.format(page_number)))
        for product in data["products"]:
            # Extract the required information from the product data and fix the list of ingredients using the Spellchecker
            product_data = {key: product[key] for key in feature_names}
            product_data["reference"] = spellchecker.predict(product["ingredients_text"])
            products_data.append(product_data)
    LOGGER.info(f"Data preparation from database finished. Dataset size: {len(products_data)}")
    return products_data


def prepare_data_from_labeled(labeled_data_path: Path) -> Iterable[Mapping]:
    """"""
    LOGGER.info("Start preparing data from manually labeled data.")
    with open(labeled_data_path, "r") as f:
        data = f.read()
    products_data = []
    for product in data.split("\n\n"):
        ingredients_text, reference, lang = product.split("\n")
        product_data = {
            "ingredients_text": ingredients_text,
            "reference": reference,
            "lang": lang
        }
        products_data.append(product_data)
    LOGGER.info(f"Data preparation from labeled data finished. Dataset size: {len(products_data)}")
    return products_data


def prepare_data_from_old_fr(
    old_fr_data_path: Path,
    sample_size: float,
    random_seed: int
) -> Iterable[Mapping]:
    """"""
    LOGGER.info("Start preparing data from old data")
    with open(old_fr_data_path, "r") as f:
        data = json.load(f)["data"]
    # We take a sample of the data
    random.seed(random_seed)
    sampled_data = random.sample(data, k=round(sample_size*len(data)))
    products_data = [
        {
            "ingredients_text":element["original"],
            "reference": element["reference"],
            "lang": element["lang"]
        } for element in sampled_data
    ]
    LOGGER.info(f"Data preparation from old data finished. Dataset size: {len(products_data)}")
    return products_data


if __name__ == "__main__":
    spellchecker = SpellChecker(
        model=OpenAIModel(
            llm=OpenAIChatCompletion(
                prompt_template=Prompt.spellcheck_prompt_template,
                system_prompt=SystemPrompt.spellcheck_system_prompt
            )
        )
    )
    prepare_benchmark(
        benchmark_version=BENCHMARK_VERSION,
        save_path=BENCHMARK_PATH,
        spellchecker=spellchecker,
        n_pages=1
    )