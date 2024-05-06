import json
from typing import Mapping, Callable, Literal
from dotenv import load_dotenv

from argilla import FeedbackDataset
from datasets import Dataset

from utils.utils import get_logger, get_repo_dir
from config.data import ArgillaConfig


load_dotenv()

LOGGER = get_logger()

REPO_DIR = get_repo_dir()
ARGILLA_DATASET_NAME = "benchmark_v4"
ARGILLA_WORKSPACE_NAME = "spellcheck"
BENCHMARK_PATH = REPO_DIR / "data/benchmark/verified_benchmark.parquet"


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
    reference = element["reference"][0]["value"] if element["reference"] else element["reference-suggestion"]
    postprocessed_reference = remove_markdown(reference)
    lang = json.loads(element["metadata"]).get("lang")
    data_origin = json.loads(element["metadata"]).get("data_origin")
    return {
        "original": element["original"],
        "reference": postprocessed_reference,
        "lang": lang,
        "data_origin": data_origin,
        "is_truncated": 0 if not element["is_truncated"] or element["is_truncated"][0]["value"] == "NO" else 1
    }
    

def postprocessing_filter_fn(
        element: Mapping, 
        status: Literal["submitted", "discarded", "draft"] = "submitted"
    ) -> bool:
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
        Mapping: True if kept, False otherwise.
    """
    # Since it can be possible there are several annotators, we only take the last annotation
    if element["reference"][0]["status"] != status:
        return False
    return True


def remove_markdown(
    text: str, 
    deleted_element: str = ArgillaConfig.deleted_element
)->  str:
    """Markdowns were added to the text in Argilla to highlight the difference with the original text. They are removed during
    the dataset extraction.

    Args:
        text (str): Text to process
        deleted_element (str, optional): To represent an element deleted from the original text.

    Returns:
        str: Post-processed text
    """
    text = text.replace("<mark>" + "~" + "</mark>", "") # Transition in deleted element: from ~ to #. Only in one in the future.
    text = text.replace("<mark>" + deleted_element + "</mark>", "")
    text = text.replace("<mark>", "").replace("</mark>", "")
    return text


if __name__ == "__main__":
    main()
