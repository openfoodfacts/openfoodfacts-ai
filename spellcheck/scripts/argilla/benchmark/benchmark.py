import os
import json
from pathlib import Path
from typing import Iterable, Dict, Mapping
from dotenv import load_dotenv

import argilla as rg
import pandas as pd

from utils.utils import get_repo_dir, show_diff

load_dotenv()

REPO_DIR = get_repo_dir()
BENCHMARK_PATH = REPO_DIR / "data/benchmark/verified_benchmark.parquet"

PRODUCT_URL = "https://world.openfoodfacts.org/product/{barcode}"
ARGILLA_DATASET_NAME = "benchmark_v4"
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
            rg.TextQuestion(name="reference", title="Correct the prediction.", use_markdown=True),
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

    benchmark = load_benchmark_parquet(BENCHMARK_PATH)
    records = prepare_records(benchmark)
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
        original = product.get("original")
        reference = product.get("reference")
        # Get product URL if available. Can be None.
        barcode = product.get("code")
        record = rg.FeedbackRecord(
            fields={
                "original": original,
                "url": PRODUCT_URL.format(barcode=barcode) if barcode else None
            },
            suggestions=[
                rg.SuggestionSchema(
                    question_name="reference",
                    value=show_diff(original, reference)
                )
            ],
            metadata={
                "lang": product.get("lang"),
                "data_origin": product.get("origin")
            }
        )
        records.append(record)
    return records


def load_benchmark_json(path: Path) -> Mapping:
    with open(path, "r") as f:
        return json.load(f)["data"]
    

def load_benchmark_parquet(path: Path) -> Mapping:
    return pd.read_parquet(path).to_dict(orient='records')

if __name__ == "__main__":  
    deploy_annotation()
    