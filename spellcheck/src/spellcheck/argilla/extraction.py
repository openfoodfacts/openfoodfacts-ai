import re
import json
from abc import ABC, abstractmethod
from typing import Mapping, Literal, List, Any

from argilla import FeedbackDataset
from pydantic import BaseModel
from datasets import Dataset

from spellcheck.utils import get_logger


LOGGER = get_logger()


class ArgillaExtraction(ABC, BaseModel):
    
    dataset_name: str
    extracted_status: List[Literal["submitted", "pending", "draft", "discarded"]]
    workspace_name: str = "spellcheck"
    deleted_element: Literal["#"] = "#"

    def extract_dataset(self) -> Dataset:

        dataset = FeedbackDataset.from_argilla(
            name=self.dataset_name, 
            workspace=self.workspace_name
        ).format_as("datasets")
        LOGGER.info(f"Dataset: {dataset}")
        processed_dataset = self._postprocess_dataset(dataset)
        LOGGER.info(f"Post-processed dataset: {processed_dataset}")
        return processed_dataset
        
    def _postprocess_dataset(self, dataset: Dataset) -> Dataset:
        return (
            dataset
            .filter(self._filter_fn)
            .map(self._map_fn, batched=False, remove_columns=dataset.column_names)
        )
    
    def _remove_highlight_markdown(self, text: str) -> str:
        """Highlights were added during Argilla deployment to show corrections.
        They are removed during the extraction.
        """
        text = re.sub("<mark(?:\s\w+[^>]*)?>" + self.deleted_element + "<\/mark>", "", text) # <mark>#</mark> - <mark style=ba...>#</mark> if an element was deleted
        text = re.sub("<\/?mark(?:\s\w+[^>]*)?>", "", text) # <mark style=ba...> - <mark> - </mark>
        return text
    
    @abstractmethod
    def _filter_fn(self, element: Mapping) -> Mapping:
        raise NotImplementedError
    
    @abstractmethod
    def _map_fn(self, element: Mapping) -> Mapping:
        raise NotImplementedError
    

class SpellcheckExtraction(ArgillaExtraction):
    """Benchmark and Training Dataset extraction.  

    Here are some examples of the extracted dataset elements during the extraction process.
    Example 1:
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

    Example 2:
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
    """

    def _map_fn(self, element: Mapping[str, Any]) -> Mapping[str, Any]:
        """Mapping function applied to Dataset with Dataset.map().
        Extract:
        * the orginal list of ingredients,
        * the suggestion from the LLM if pending or the annotator's correction,
        * the language
        * and the "data origin" if avlaible (Legacy, will be remmoved in the future)


        Args:
            element (Mapping[str, Any]): Dictionnary containing the feature as key and value

        Returns:
            Mapping[str, Any]: Processed element.
        """
        # If status pending, we take the suggestion from the LLM
        reference = element["reference"][0]["value"] if element["reference"] else element["reference-suggestion"]
        postprocessed_reference = self._remove_highlight_markdown(reference)
        # Metadata is JSON encoded
        lang = json.loads(element["metadata"]).get("lang")
        return {
            "original": element["original"],
            "reference": postprocessed_reference,
            "lang": lang,
            "code": element.get("code"),
            "is_truncated": 0 if not element.get("is_truncated") or element["is_truncated"][0]["value"] == "NO" else 1
        }
    
    def _filter_fn(self, element: Mapping[str, Any]) -> bool:
        """Filter function applied to Dataset with Dataset.filter()

        Args:
            element (Mapping[str, Any]): Dictionnary containing the feature as key and value

        Returns:
            bool: whether to keep (True) or drop (False) the element
        """
        reference = element.get("reference")
        # Status == Pending means no annotation were performed by annotator, but the LLM suggestion remains. 
        if not reference and "pending" in self.extracted_status:
            return True
        # Since it can be possible there are several annotators, we only take the last annotation
        if reference and reference[0]["status"] in self.extracted_status:
            return True
        return False
    

class SpellcheckDPOExtraction(ArgillaExtraction):
    """Extract chosen and rejected correction from Argilla. This dataset is used to train a DPO (Direct Preference Optimization) model.

    * 'Chosen': annotator modification.
    * 'Rejected': LLM original suggestion
    """
    
    def __init__(self, **kwargs) -> None:
        """DPO Extraction only works for "submitted" are in extracted_status.
        """
        super().__init__(**kwargs)
        if "submitted" not in self.extracted_status:
            raise ValueError(f"'Submitted' not in extracted_status. Current status: {self.extracted_status}")

    def _map_fn(self, element: Mapping[str, Any]) -> Mapping[str, Any]:
        """_summary_

        Args:
            element (Mapping[str, Any]): _description_

        Returns:
            Mapping[str, Any]: _description_
        """
        # If status pending, we take the suggestion from the LLM
        chosen = element["reference"][0]["value"]
        rejected = element["reference-suggestion"]
        postprocessed_chosen = self._remove_highlight_markdown(chosen)
        postprocessed_rejected = self._remove_highlight_markdown(rejected)
        # Metadata is JSON encoded
        lang = json.loads(element["metadata"]).get("lang")
        return {
            "original": element["original"],
            "chosen": postprocessed_chosen,
            "rejected": postprocessed_rejected,
            "lang": lang,
        }
    
    def _filter_fn(self, element: Mapping[str, Any]) -> bool:
        """Filter function that considers only examples with submitted annotations.

        Args:
            element (Mapping[str, Any]): Dataset row

        Returns:
            bool: Whether the row is kept.
        """
        reference = element.get("reference")
        if not reference: 
            return False
        if reference[0]["status"] in self.extracted_status:
            return True
        return False 