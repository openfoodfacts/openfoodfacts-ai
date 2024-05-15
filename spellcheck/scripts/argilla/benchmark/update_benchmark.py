import os
import logging
from datetime import datetime
from dotenv import load_dotenv
from typing import Iterable

import argilla as rg

from spellcheck.utils import show_diff


logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
LOGGER = logging.getLogger(__name__)

ARGILLA_TEST_DATASET_NAME = "benchmark_v2"
ARGILLA_DATASET_NAME = "benchmark"
ARGILLA_WORKSPACE_NAME = "spellcheck"

load_dotenv()


def main():
    push_dataset_updates(
        previous_name=ARGILLA_DATASET_NAME,
        workspace=ARGILLA_WORKSPACE_NAME,
        new_name=ARGILLA_TEST_DATASET_NAME
    )


def update_dataset(name: str, workspace: str) -> None:
    """Update exisiting Argilla dataset. 
    Only suggestions, responses and metadata can be modified.
    ATTENTION: test the update by creating a new dataset first to ensure you don't delete existing annotations.

    Args: 
        name (str): Dataset name
        workspace (str): Workspace name
    """
    # Extract previous annotation
    dataset = rg.FeedbackDataset.from_argilla(
        name=name, 
        workspace=workspace
    )
    updated_records = update_records(dataset.records)
    dataset.update_records(updated_records)
    

def update_records(records: Iterable[rg.FeedbackRecord]) -> Iterable[rg.FeedbackRecord]:
    """Update records.

    Args:
        records (Iterable[Dict[str, str]]): Existing Feedback records

    Returns:
        Iterable[rg.FeedbackRecord]: Modifed records.
    """
    modified_record = []
    for record in records:
        original = record.fields.get("original")
        suggestion = record.suggestions[0].value if record.suggestions else None
        response = record.responses[0].values["reference"].value if record.responses else None
        if suggestion:
            record.suggestions = [
                {
                    "question_name": "reference",
                    "value": show_diff(original, suggestion),
                    "agent": "gpt-3.5"
                }
            ]
        if response:
            record.responses = [
                    {
                        "values":{
                            "reference":{
                                "value": show_diff(original, response),
                            }
                        },
                        "inserted_at": datetime.now(),
                        "updated_at": datetime.now(),
                        "status": "submitted"
                    }
                ]
        record.metadata = {
                "lang": record.metadata.get("lang"),
                "data_origin": _update_metadata(record.metadata["data_origin"])
            }
        modified_record.append(record)
    return modified_record


def _update_metadata(metadata: str) -> str:
    """Modify existing metadata"""
    if metadata == "old_data":
        return "manually_labeled_french_list"
    elif metadata == "labeled_data":
        return "manually_labeled_multi_lang_list"
    else:
        return metadata


def push_dataset_updates(previous_name: str, workspace: str, new_name: str) -> None:
    """Push new dataset instead of updating previous one.
    Ensure no data is lost during the process.

    Args:
        previous_name (str): Dataset to get the data from
        workspace (str): Workspace name
        new_name (str): New dataset name
    """
    rg.init(
        api_url=os.getenv("ARGILLA_API_URL"),
        api_key=os.getenv("ARGILLA_API_KEY")
    )

    dataset = rg.FeedbackDataset(
        fields=[
            rg.TextField(name="url", title="Product URL", required=False),
            rg.TextField(name="original", title="Original", use_markdown=True),
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

    # Load previous dataset
    previous_dataset = rg.FeedbackDataset.from_argilla(
        name=previous_name, 
        workspace=workspace
    )
    updated_records = update_records(previous_dataset.records)
    dataset.add_records(updated_records)
    dataset.push_to_argilla(name=new_name, workspace=workspace)



if __name__ == "__main__":
    main()