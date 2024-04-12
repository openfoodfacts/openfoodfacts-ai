import os
import logging
import difflib
from dotenv import load_dotenv
from typing import Iterable

import argilla as rg


logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
LOGGER = logging.getLogger(__name__)

ARGILLA_DATASET_NAME = "benchmark_with_highlight"
ARGILLA_PREVIOUS_DATASET_NAME = "benchmark"
ARGILLA_WORKSPACE_NAME = "spellcheck"

load_dotenv()


def deploy_dataset(name: str, workspace: str, previous_name: str):
    """_summary_
    """
    # rg.init(
    #     api_url=os.getenv("ARGILLA_API_URL"),
    #     api_key=os.getenv("ARGILLA_API_KEY")
    # )

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

    # Extract previous annotation
    previous_dataset = rg.FeedbackDataset.from_argilla(
        name=previous_name, 
        workspace=workspace
    )
    records = [record for record in previous_dataset.records]
    updated_records = update_records(records=records)
    dataset.add_records(records=updated_records)
    # dataset.push_to_argilla(name, workspace)
    

def update_records(records: Iterable[rg.FeedbackRecord]) -> Iterable[rg.FeedbackRecord]:
    """_summary_

    Args:
        data (Iterable[Dict[str, str]]): _description_

    Returns:
        Iterable[rg.FeedbackRecord]: _description_
    """
    updated_records = []
    for record in records:
        # Extract texts from record
        original = record.fields.get("original")
        suggestion = record.suggestions[0].value
        response = record.responses[0].values["reference"].value

        record = rg.FeedbackRecord(
            fields={
                "original": original,
                "url": record.fields.get("url")
            },
            suggestions=[
                rg.SuggestionSchema(
                    question_name="reference",
                    value=_highlight_text(original, suggestion)
                )
            ],
            metadata={
                "lang": record.metadata.get("lang"),
                "data_origin": _update_metadata(record.metadata["data_origin"])
            }
        )
        updated_records.append(record)
    return updated_records


# def _update_records(records: Iterable[rg.FeedbackRecord]) -> Iterable[rg.FeedbackRecord]:

#     modified_records = []
#     for record in records:
#         original_text = record.fields["original"]
#         suggestion_text = record.suggestions[0].value
#         # response = record.responses[0].values["reference"].value

#         highlighted_suggestion = _highlight_text(original=original_text, text_to_highlight=suggestion_text)

#         #Update record
#         #Need
#         list(record.suggestions)[0].value = highlighted_suggestion
#         record.responses[0].values["reference"].value = highlighted_response

#         # Update metadata to add more details about the origin of the data
#         if record.metadata["data_origin"] == "old_data":
#             record.metadata["data_origin"] = "manually_labeled_french_list"
#         elif record.metadata["data_origin"] == "labeled_data":
#             record.metadata["data_origin"] = "manually_labeled_multi_lang_list"
        
#         modified_records.append(record)
#     return modified_records


def _update_metadata(metadata: str) -> str:
    """"""
    if metadata == "old_data":
        return "manually_labeled_french_list"
    elif metadata == "labeled_data":
        return "manually_labeled_multi_lang_list"
    else:
        return metadata
    

def _highlight_text(original: str, text_to_highlight: str, highlight_mark: str = "<mark>") -> str:
    """Highlight any work in suggestion that is different from the original.
    To be highlighted in Argilla, the word needs to be enclosed by "<mark>".

    Args:
        original (str): Original text to correct.
        suggestion (str): Agent suggestion.
        highlight_mark (str, optional): Highlight recognized by markdown. Defaults to "<mark>".

    Returns:
        str: Marked text
    """
    differ = difflib.Differ()
    diff = list(differ.compare(original, text_to_highlight))  # Output ['  I', '   ', '  l', '  o', '  v', '- e', '   ', '  p', '  e', '  a', '  r', '  s', '+ o', '+ n']

    marked_text = ""
    mark = "<marked>" # Mark to identify words with differences
    for item in diff:
        if item.startswith("-") or item.startswith("+"):
            marked_text += mark + item[2:]
        elif item.startswith(" "):
            marked_text += item[2:]

    # Identify marked words and enclose them with highlight_mark
    highlighted_text = " ".join(
        [
            highlight_mark + token.replace(mark, "") + highlight_mark
            if mark in token
            else token
            for token in marked_text.split()
        ]
    )
    return highlighted_text


if __name__ == "__main__":
    deploy_dataset(
        name=ARGILLA_DATASET_NAME,
        workspace=ARGILLA_WORKSPACE_NAME,
        previous_name=ARGILLA_PREVIOUS_DATASET_NAME
    )