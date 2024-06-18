from typing import Iterable

import argilla as rg
import pandas as pd 

from spellcheck.utils import get_logger, get_repo_dir, show_diff


LOGGER = get_logger()

REPO_DIR = get_repo_dir()
DATA_PATH = REPO_DIR / "data/benchmark/additional_products/synthetically_corrected_products.parquet"

ARGILLA_DATASET_NAME = "benchmark_v2"
ARGILLA_WORKSPACE_NAME = "spellcheck"

PRODUCT_URL = "https://world.openfoodfacts.org/api/v2/product/{}"


def main():
    
    df = pd.read_parquet(DATA_PATH)
    LOGGER.info(f"Features: {df.columns}")
    dataset = rg.FeedbackDataset.from_argilla(
        name=ARGILLA_DATASET_NAME, 
        workspace=ARGILLA_WORKSPACE_NAME
    )
    records = prepare_records(
        originals=df["ingredients_text"].tolist(),
        references=df["correction"].tolist(),
        codes=df["code"].tolist(),
        langs=df["lang"].tolist()
    )
    dataset.add_records(records)


def prepare_records(
    originals: Iterable[str],
    references: Iterable[str],
    langs: Iterable[str],
    codes: Iterable[int]
) -> Iterable[rg.FeedbackRecord]:
    """Prepare records for Argilla.

    Args:
        originals (Iterable[str]): Original lists of ingredients
        references (Iterable[str]): Correction suggested by the LLM agent
        langs (Iterable[str]): Languages
        codes (Iterable[int]): Barcodes

    Returns:
        Iterable[rg.FeedbackRecord]: Records.
    """
    records = [
        rg.FeedbackRecord(
            fields={
                "original": original,
                "url": PRODUCT_URL.format(code) if code else None
            },
            suggestions=[
                rg.SuggestionSchema(
                    question_name="reference",
                    value=show_diff(original_text=original, corrected_text=reference)
                )
            ],
            metadata={
                "lang": lang
            }
        ) for original, reference, code, lang in zip(originals, references, codes, langs)
    ]
    return records


if __name__ =="__main__":
    main()
