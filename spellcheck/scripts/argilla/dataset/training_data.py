import os
from typing import Iterable, Mapping
from dotenv import load_dotenv

import argilla as rg

from utils.utils import show_diff, get_repo_dir, load_jsonl


load_dotenv()

REPO_DIR = get_repo_dir()
SYNTHETIC_DATASET = REPO_DIR / "data/dataset/synthetic_data.jsonl"

ARGILLA_DATASET_NAME = "training_dataset"
ARGILLA_WORKSPACE_NAME = "spellcheck"


def deploy_annotation():
    """Deploy Argilla annotation tool to OFF servers.
    """
    rg.init(
        api_url=os.getenv("ARGILLA_API_URL"),
        api_key=os.getenv("ARGILLA_API_KEY")
    )

    dataset = rg.FeedbackDataset(
        fields=[
            rg.TextField(name="original", title="Original", use_markdown=True),
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
            rg.TermsMetadataProperty(name="lang", title="Language")
        ],
    )

    data = load_jsonl(path=SYNTHETIC_DATASET)
    records = prepare_records(data)
    dataset.add_records(records=records)
    dataset.push_to_argilla(name=ARGILLA_DATASET_NAME, workspace=ARGILLA_WORKSPACE_NAME)
    

def prepare_records(data: Iterable[Mapping]) -> Iterable[rg.FeedbackRecord]:
    """Prepare Feedback records.

    Note:
        Difference between original texts and suggestions are highlighted using the show_diff(). 

    Args:
        data (Iterable[Dict[str, str]]): Data containing texts to correct and suggestions from 
    OpenAI agent.

    Returns:
        Iterable[rg.FeedbackRecord]: Records
    """
    records = []
    for product in data:
        original_text = product.get("ingredients_text")
        corrected_text = product.get("corrected_text")

        record = rg.FeedbackRecord(
            fields={
                "original": original_text,
            },
            suggestions=[
                rg.SuggestionSchema(
                    question_name="reference",
                    value=show_diff(
                        original_text=original_text,
                        corrected_text=corrected_text
                    )
                )
            ],
            metadata={
                "lang": product.get("lang"),
            }
        )
        records.append(record)
    return records


if __name__ == "__main__":  
    deploy_annotation()
    