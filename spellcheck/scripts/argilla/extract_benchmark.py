import os
import logging
import json
from pathlib import Path
from typing import Mapping, Callable, Literal
from dotenv import load_dotenv

from argilla import FeedbackDataset
from datasets import Dataset


load_dotenv()

logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
LOGGER = logging.getLogger(__name__)

SPELLCHECK_DIR = Path(os.path.realpath(__file__)).parent.parent.parent
ARGILLA_DATASET_NAME = "benchmark"
ARGILLA_WORKSPACE_NAME = "spellcheck"
BENCHMARK_PATH = SPELLCHECK_DIR / "data/benchmark/verified_benchmark.parquet"


def main():
    "Main function"
    dataset = extract_dataset(
        name=ARGILLA_DATASET_NAME,
        workspace=ARGILLA_WORKSPACE_NAME,
        postprocess_map_fn=postprocessing_map_fn,
        postprocess_filter_fn=postprocessing_filter_fn
    )
    dataset.to_parquet(BENCHMARK_PATH)


def extract_dataset(
        name: str, 
        workspace: str, 
        postprocess_map_fn: Callable = None, 
        postprocess_filter_fn: Callable = None
    ) -> Dataset:
    """Extract the annotated dataset from the deployed Argilla. 

    Args:
        name (str): Argilla dataset
        workspace (str): Argilla workspace
        postprocess_map_fn (Callable): Dataset postprocessing function
        postprocess_filter_fn (Callable): Dataset filtering function

    Returns:
        Dataset: Extracted dataset in the Hugging Face Dataset format
    """
    dataset = FeedbackDataset.from_argilla(name=name, workspace=workspace)
    hf_dataset = dataset.format_as("datasets")
    LOGGER.info(f"Dataset: {hf_dataset}")
    if postprocess_map_fn:
        return postprocess_dataset(
            dataset=hf_dataset, 
            map_function=postprocess_map_fn,
            filter_fn=postprocess_filter_fn
        )
    return hf_dataset


def postprocess_dataset(dataset: Dataset, map_function: Callable, filter_fn: Callable) -> Dataset:
    """Post-processing the dataset. 

    Args:
        dataset (Dataset): Exported dataset from Argilla
        map_function (Callable): Function applied to the dataset
        filter_fn (Callable): Filter function

    Returns:
        Dataset: Post-processed dataset
    """
    dataset = dataset.filter(function=filter_fn)
    dataset = dataset.map(function=map_function, remove_columns=dataset.column_names)
    return dataset


def postprocessing_map_fn(element: Mapping) -> Mapping:
    """Mapping unction applied to the dataset.

    Args:
        element (Mapping): 
            One row of the extracted dataset before processing:

            ```
            'url': None
            'original': 'Ananas, Ananassaft, Säuerungs - mittel: Citronensäure'
            'reference': [
                {
                'user_id': 'dfb71753-1187-45e1-8006-629bef2b49e0', 
                'value': 'Ananas, Ananassaft, Säuerungsmittel: Citronensäure', 
                'status': 'submitted'
                }
            ]
            'reference-suggestion': 'Ananas, Ananassaft, Säuerungsmittel: Citronensäure'
            'reference-suggestion-metadata': {'type': None, 'score': None, 'agent': None}
            'is_truncated': [{'user_id': 'dfb71753-1187-45e1-8006-629bef2b49e0', 'value': 'NO', 'status': 'submitted'}]
            'is_truncated-suggestion': None
            'is_truncated-suggestion-metadata': {'type': None, 'score': None, 'agent': None}
            'external_id': None
            'metadata': '{"lang": "de", "data_origin": "labeled_data"}'
            ```
            
    Returns:
        Mapping: Post-processed element
    """
    return {
        "original": element["original"],
        "reference": element["reference"][0]["value"] if element["reference"] else element["reference-suggestion"],
        "lang": json.loads(element["metadata"])['lang'],
        "data_origin": json.loads(element["metadata"])["data_origin"],
        "is_truncated": 0 if not element["is_truncated"] or element["is_truncated"][0]["value"] == "NO" else 1
    }
    

def postprocessing_filter_fn(
        element: Mapping, 
        status: Literal["submitted", "discarded"] = "discarded"
    ) -> Mapping:
    """Filter dataset depending on annotation status.

    Args:
        element (Mapping): 
            One row of the extracted dataset before processing 
            (One would notice that this data was 'discarded' by the annotator)

            ```
            'url': 'https://world.openfoodfacts.org/product/5942262001416'
            'original': 'water:snow' 
            'reference': [{'user_id': 'dfb71753-1187-45e1-8006-629bef2b49e0', 'value': 'water:snow', 'status': 'discarded'}]
            'reference-suggestion': 'water:snow'
            'reference-suggestion-metadata': {'type': None, 'score': None, 'agent': None}
            'is_truncated': []
            'is_truncated-suggestion': None
            'is_truncated-suggestion-metadata': {'type': None, 'score': None, 'agent': None}
            'external_id': None
            'metadata': '{"lang": "ro", "data_origin": "50-percent-unknown"}'
            ```
        status (Literal["submitted", "discarded"], optional): Annotation status to filter. Defaults to "discarded".

    Returns:
        Mapping: _description_
    """
    # Since it can be possible there are several annotators, we only take the last annotation
    for reference in element["reference"]:
        if reference["status"] == status:
            return False
        break 
    return True


if __name__ == "__main__":
    main()
