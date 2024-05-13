import os
import json
from abc import ABC, abstractmethod
from typing import Iterable, Dict
from pathlib import Path
from dotenv import load_dotenv

import argilla as rg

from utils.utils import show_diff


load_dotenv()


class ArgillaModule(ABC):
    """Class to prepare datasets and interact with Argilla."""

    @abstractmethod
    def deploy():
        """Deploy the dataset into Argilla."""
        raise NotImplementedError
    
    @abstractmethod
    def _prepare_records(self) -> Iterable[rg.FeedbackRecord]:
        """Records are prepared in respect of the preconfigured fields. 

        Returns:
            Iterable[rg.FeedbackRecord]: Batch of records. 
        """
        raise NotImplementedError
    
    @classmethod
    @abstractmethod
    def from_jsonl(path: Path) -> None:
        """Load the data from a JSONL file."""
        raise NotImplementedError
    

class BenchmarkEvaluationArgilla(ArgillaModule):
    """Argilla module for model visual evaluation.
    
    Args:
            originals (Iterable[str]): Batch of original lists of ingredients
            references (Iterable[str]): Batch of references as annotated in the benchmark
            predictions (Iterable[str]): Batch of model predictions
            metadata (Iterable[Dict]): Batch of metadata associated with each list of ingredients
    """
    def __init__(
        self,
        originals: Iterable[str],
        references: Iterable[str],
        predictions: Iterable[str],
        metadata: Iterable[Dict]
    ):
        self.originals = originals
        self.references = references
        self.predictions = predictions
        self.metadata = metadata

    def deploy(
        self, 
        dataset_name: str, 
        workspace_name: str = "spellcheck"
    ) -> None:
        """Deploy lists of ingredients into Argilla for visual evaluation. 
        Modifications are highlighted for assisting the annotator.
        The annotator decides if the model prediction is valid or not.

        Args:
            dataset_name (str): Argilla dataset name
            workspace_name (str, optional): Argilla workspace name. Defaults to "spellcheck".
        """
        rg.init(
            api_url=os.getenv("ARGILLA_API_URL"),
            api_key=os.getenv("ARGILLA_API_KEY")
        )
        dataset = rg.FeedbackDataset(
            fields=[
                rg.TextField(name="original", title="Original", use_markdown=True),
                rg.TextField(name="reference", title="Reference", use_markdown=True),
                rg.TextField(name="prediction", title="Prediction", use_markdown=True)
            ],
            questions=[
                rg.LabelQuestion(
                    name="is_good",
                    title="Is the correction correct?",
                    labels=["Good","Bad"],
                    required=True
                ),
                rg.TextQuestion(
                    name="notes",
                    title="Explain your decision:   ",
                    required=False
                )
            ],
            metadata_properties=[
                rg.TermsMetadataProperty(name="lang", title="Language"),
            ],
        )
        records = self._prepare_records()
        dataset.add_records(records=records)
        dataset.push_to_argilla(name=dataset_name, workspace=workspace_name)
            

    def _prepare_records(self) -> Iterable[rg.FeedbackRecord]:
        """Records are prepared in respect of the preconfigured fields. 

        Returns:
            Iterable[rg.FeedbackRecord]: Batch of records. 
        """
        records = []
        for original, reference, prediction, metadata in zip(
            self.originals, self.highlighted_references, self.highlighted_predictions, self.metadata
        ):
            record = rg.FeedbackRecord(
                fields={
                    "original": original,
                    "reference": reference,
                    "prediction": prediction
                },
                metadata={
                    "lang": metadata["lang"],
                }
            )
            records.append(record)
        return records
    
    @classmethod
    def from_jsonl(cls, path: Path):
        with open(path, 'r') as file:
            elements =  [json.loads(line) for line in file.readlines()]
        return cls(
            [element["original"] for element in elements],
            [element["reference"] for element in elements],
            [element["prediction"] for element in elements],
            [element["metadata"] for element in elements]
        )
    
    @property
    def highlighted_references(self):
        """Highlight references.
        """
        return [show_diff(original, reference, color="yellow") for original, reference in zip(self.originals, self.references)]

    @property
    def highlighted_predictions(self):
        """Highlight predictions.
        """
        return [show_diff(reference, prediction, color="red") for reference, prediction in zip(self.references, self.predictions)]


class IngredientsCompleteEvaluationArgilla(ArgillaModule):
    """Prepare Ingredients-Complete dataset for False Positives verification."""

    def __init__(
        self, 
        originals: Iterable[str], 
        predictions: Iterable[str],
        metadata: Iterable[Dict]
    ):
        self.originals = originals
        self.predictions = predictions
        self.metadata = metadata
    
    def deploy(
        self, 
        dataset_name: str, 
        workspace_name: str = "spellcheck"
    ) -> None:
        """Deploy lists of ingredients into Argilla for visual evaluation. 
        Modifications are highlighted for assisting the annotator.
        The annotator decides if the model prediction is valid or not.

        Args:
            dataset_name (str): Argilla dataset name
            workspace_name (str, optional): Argilla workspace name. Defaults to "spellcheck".
        """
        rg.init(
            api_url=os.getenv("ARGILLA_API_URL"),
            api_key=os.getenv("ARGILLA_API_KEY")
        )
        dataset = rg.FeedbackDataset(
            fields=[
                rg.TextField(name="original", title="Original", use_markdown=True),
                rg.TextField(name="prediction", title="Prediction", use_markdown=True)
            ],
            questions=[
                rg.LabelQuestion(
                    name="is_good",
                    title="Is the correction correct?",
                    labels=["Good","Bad"],
                    required=True
                ),
                rg.TextQuestion(
                    name="notes",
                    title="Explain your decision:   ",
                    required=False
                )
            ],
            metadata_properties=[
                rg.TermsMetadataProperty(name="lang", title="Language"),
                rg.TermsMetadataProperty(name="code", title="Code")
            ],
        )
        records = self._prepare_records()
        dataset.add_records(records=records)
        dataset.push_to_argilla(name=dataset_name, workspace=workspace_name)
            

    def _prepare_records(self) -> Iterable[rg.FeedbackRecord]:
        """Records are prepared in respect of the preconfigured fields. 

        Returns:
            Iterable[rg.FeedbackRecord]: Batch of records. 
        """
        records = []
        for original, prediction, metadata in zip(
            self.originals, self.highlighted_predictions, self.metadata
        ):
            record = rg.FeedbackRecord(
                fields={
                    "original": original,
                    "prediction": prediction
                },
                metadata={
                    "lang": metadata["lang"],
                    "code": metadata["code"]
                }
            )
            records.append(record)
        return records
    
    @classmethod
    def from_jsonl(cls, path: Path):
        with open(path, 'r') as file:
            elements =  [json.loads(line) for line in file.readlines()]
        return cls(
            [element["original"] for element in elements],
            [element["prediction"] for element in elements],
            [element["metadata"] for element in elements]
        )
    
    @property
    def highlighted_predictions(self):
        """Highlight predictions.
        """
        return [show_diff(original, prediction, color="red") for original, prediction in zip(self.originals, self.predictions)]
