import os
import json
from pathlib import Path
from typing import Iterable, Dict
from dotenv import load_dotenv

import argilla as rg


load_dotenv()

SPELLCHECK_DIR = Path(os.path.realpath(__file__)).parent.parent.parent
UNCHECKED_BENCHMARK_PATH = SPELLCHECK_DIR / "data/benchmark/benchmark.json"

PRODUCT_URL = "https://world.openfoodfacts.org/product/{barcode}"
ARGILLA_DATASET_NAME = "benchmark"
ARGILLA_WORKSPACE_NAME = "spellcheck"


def deploy_annotation():
    """_summary_
    """
    rg.init(
        api_url=os.getenv("ARGILLA_API_URL"),
        api_key=os.getenv("ARGILLA_API_KEY")
    )

    dataset = rg.FeedbackDataset(
        fields=[
            rg.TextField(name="url", title="Product URL", required=False),
            rg.TextField(name="original", title="Original"),
        ],
        questions=[
            rg.TextQuestion(name="reference", title="Correct the prediction."),
            rg.LabelQuestion(
                name="is_truncated",
                title="Is the list of ingredients truncated?",
                labels=["YES","NO"],
                required=False
            )
        ],
        metadata_properties=[
            rg.TermsMetadataProperty(name="lang", title="Language"),
            rg.TermsMetadataProperty(name="data_origin", title="Origin")
        ],
    )

    # Prepare records from benchmark data
    with open(UNCHECKED_BENCHMARK_PATH, "r") as f:
        benchmark = json.load(f)
    records = prepare_records(benchmark["data"])

    dataset.add_records(records=records)
    dataset.push_to_argilla(name=ARGILLA_DATASET_NAME, workspace=ARGILLA_WORKSPACE_NAME)
    

def prepare_records(data: Iterable[Dict[str, str]]) -> Iterable[rg.FeedbackRecord]:
    """_summary_

    Args:
        data (Iterable[Dict[str, str]]): _description_

    Returns:
        Iterable[rg.FeedbackRecord]: _description_
    """
    records = []
    for product in data:
        # Get product URL if available. Can be None.
        barcode = product.get("code")
        record = rg.FeedbackRecord(
            fields={
                "original": product.get("ingredients_text"),
                "url": PRODUCT_URL.format(barcode=barcode) if barcode else None
            },
            suggestions=[
                rg.SuggestionSchema(
                    question_name="reference",
                    value=product.get("reference")
                )
            ],
            metadata={
                "lang": product.get("lang"),
                "data_origin": product.get("origin")
            }
        )
        records.append(record)
    return records


if __name__ == "__main__":  
    deploy_annotation()
    