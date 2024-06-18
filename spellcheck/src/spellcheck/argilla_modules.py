import os
from abc import ABC, abstractmethod
from typing import Iterable, Dict
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm

import argilla as rg
import pandas as pd
import datasets

from spellcheck.utils import show_diff, load_jsonl


load_dotenv()


class ArgillaModule(ABC):
    """Class to prepare datasets and interact with Argilla."""

    @abstractmethod
    def deploy(self, dataset_name: str, workspace_name: str = "spellcheck") -> None:
        """Deploy Dataset into Argilla. 

        Args:
            dataset_name (str): Argilla dataset name
            workspace_name (str, optional): Argilla workspace name. Defaults to "spellcheck".
        """
        raise NotImplementedError
    
    @abstractmethod
    def _prepare_dataset(self) -> rg.FeedbackDataset:
        """Prepare Argilla Dataset architecture for annotation.

        Returns:
            rg.FeedbackDataset
        """
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
    def from_jsonl(cls, path: Path) -> None:
        """Load the data from a JSONL file."""
        raise NotImplementedError
    
    @classmethod
    @abstractmethod
    def from_parquet(cls, path: Path) -> None:
        """Load the data from a parquet file."""
        raise NotImplementedError
    
    @classmethod
    @abstractmethod
    def from_s3(cls, uri) -> None:
        """Load from an S3 uri. The S3 uri needs to lead to a HF dataset folder."""
        raise NotImplementedError


class BenchmarkEvaluationArgilla(ArgillaModule):
    """Argilla module for model human evaluation step.
    
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
        rg.init(
            api_url=os.getenv("ARGILLA_API_URL"),
            api_key=os.getenv("ARGILLA_API_KEY")
        )
        dataset = self._prepare_dataset()
        records = self._prepare_records()
        dataset.add_records(records=records)
        dataset.push_to_argilla(name=dataset_name, workspace=workspace_name)
            

    def _prepare_dataset(self) -> rg.FeedbackDataset:
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
        return dataset
    
    def _prepare_records(self) -> Iterable[rg.FeedbackRecord]:
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
                    "lang": metadata.get("lang"),
                }
            )
            records.append(record)
        return records
    
    @classmethod
    def from_jsonl(cls, path: Path):
        elements = load_jsonl(path)
        return cls(
            [element["original"] for element in elements],
            [element["reference"] for element in elements],
            [element["prediction"] for element in elements],
            [element["metadata"] for element in elements]
        )
    
    @classmethod
    def from_parquet(path: Path) -> None:
        raise NotImplementedError
    
    @classmethod
    def from_s3(cls, uri: str) -> None:
        if os.path.splitext(uri)[-1]:
            raise ValueError("The S3 uri should be directed to a Hugging Face Dataset folder.")
        dataset = datasets.load_from_disk(uri)
        return cls(
            dataset["original"],
            dataset["reference"],
            dataset["prediction"],
            [{"lang": lang} for lang in dataset["lang"]]
        )
    
    @property
    def highlighted_references(self) -> Iterable[str]:
        """Highlight references.
        """
        return [show_diff(original, reference, color="yellow") for original, reference in zip(self.originals, self.references)]

    @property
    def highlighted_predictions(self) -> Iterable[str]:
        """Highlight predictions.
        """
        return [show_diff(reference, prediction, color="red") for reference, prediction in zip(self.references, self.predictions)]
    

class IngredientsCompleteEvaluationArgilla(ArgillaModule):
    """Prepare Ingredients-Complete dataset for False Positives verification.
    
    Args:
        originals (Iterable[str]): Batch of original lists of ingredients
        predictions (Iterable[str]): Batch of model predictions
        metadata (Iterable[Dict]): Batch of metadata associated with each list of ingredients
    """
    
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
        rg.init(
            api_url=os.getenv("ARGILLA_API_URL"),
            api_key=os.getenv("ARGILLA_API_KEY")
        )
        dataset = self._prepare_dataset()
        records = self._prepare_records()
        dataset.add_records(records=records)
        dataset.push_to_argilla(name=dataset_name, workspace=workspace_name)
            
    def _prepare_dataset(self) -> rg.FeedbackDataset:
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
        return dataset
    
    def _prepare_records(self) -> Iterable[rg.FeedbackRecord]:
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
                    "lang": metadata.get("lang"),
                    "code": str(metadata.get("code")) # String required instead of int
                }
            )
            records.append(record)
        return records
    
    @classmethod
    def from_jsonl(cls, path: Path):
        elements = load_jsonl(path)
        return cls(
            [element["original"] for element in elements],
            [element["prediction"] for element in elements],
            [element["metadata"] for element in elements]
        )
    
    @classmethod
    def from_parquet(path: Path) -> None:
        raise NotImplementedError
    
    @classmethod
    def from_s3(path: Path) -> None:
        raise NotImplementedError

    @property
    def highlighted_predictions(self):
        """Highlight predictions.
        """
        return [show_diff(original, prediction, color="red") for original, prediction in zip(self.originals, self.predictions)]


class BenchmarkArgilla(ArgillaModule):
    """Generate Benchmark annotation on Argilla.

    Args:
        originals (Iterable[str]): Batch of original lists of ingredients
        references (Iterable[str]): Batch of references as annotated in the benchmark
        metadata (Iterable[Dict]): Batch of metadata associated with each list of ingredients
    """

    def __init__(
        self,
        originals: Iterable[str],
        references: Iterable[str],
        metadata: Iterable[Dict],
    ) -> None:
        self.originals = originals
        self.references = references
        self.metadata = metadata

    def deploy(
        self,
        dataset_name: str, 
        workspace_name: str = "spellcheck"
    ):
        rg.init(
            api_url=os.getenv("ARGILLA_API_URL"),
            api_key=os.getenv("ARGILLA_API_KEY")
        )
        dataset = self._prepare_dataset()
        records = self._prepare_records()
        dataset.add_records(records=records)
        dataset.push_to_argilla(name=dataset_name, workspace=workspace_name)

    def _prepare_dataset(self) -> rg.FeedbackDataset:
        dataset = rg.FeedbackDataset(
            fields=[
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
            ],
        )
        return dataset
        
    def _prepare_records(self):
        records = []
        for original, reference, metadata in zip(self.originals, self.references, self.metadata):
            record = rg.FeedbackRecord(
                fields={
                    "original": original,
                },
                suggestions=[
                    rg.SuggestionSchema(
                        question_name="reference",
                        value=show_diff(original, reference)
                    )
                ],
                metadata={
                    "lang": metadata.get("lang"),
                }
            )
            records.append(record)
        return records
    
    @classmethod
    def from_parquet(cls, path: Path):
        """Load the data from a parquet file."""
        df = pd.read_parquet(path)
        metadata = [{"lang": lang} for lang in df["lang"]]
        return cls(
            originals=df["original"].tolist(),
            references=df["reference"].tolist(),
            metadata=metadata
        )
    
    @classmethod
    def from_jsonl(path: Path) -> None:
        raise NotImplementedError
    
    @classmethod
    def from_s3(path: Path) -> None:
        raise NotImplementedError


class TrainingDataArgilla(ArgillaModule):
    """Argilla module for deploying training dataset for annotation."""

    def __init__(
        self,
        originals: Iterable[str],
        references: Iterable[str],
        metadata: Iterable[Dict]
    ) -> None:
        self.originals = originals
        self.references = references
        self.metadata = metadata

    def deploy(self, dataset_name: str, workspace_name: str = "spellcheck"):
        rg.init(
            api_url=os.getenv("ARGILLA_API_URL"),
            api_key=os.getenv("ARGILLA_API_KEY")
        )
        dataset = self._prepare_dataset()
        records = self._prepare_records()
        dataset.add_records(records)
        dataset.push_to_argilla(name=dataset_name, workspace=workspace_name)

    def _prepare_dataset(self) -> rg.FeedbackDataset:
        dataset = rg.FeedbackDataset(
            fields=[
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
                rg.TermsMetadataProperty(name="lang", title="Language")
            ],
        )
        return dataset
    
    def _prepare_records(self) -> Iterable[rg.FeedbackRecord]:
        records = []
        for original, highlighted_reference, metadata in zip(
            self.originals, 
            self.highlighted_references, 
            self.metadata
        ):
            record = rg.FeedbackRecord(
                fields={
                    "original": original,
                },
                suggestions=[
                    rg.SuggestionSchema(
                        question_name="reference",
                        value=highlighted_reference
                    )
                ],
                metadata={
                    "lang": metadata.get("lang"),
                }
            )
            records.append(record)
        return records
    
    @property
    def highlighted_references(self):
        return [show_diff(original, reference) for original, reference in tqdm(zip(self.originals, self.references))]

    @classmethod
    def from_jsonl(cls, path: Path):
        elements = load_jsonl(path)
        return cls(
            [element["original"] for element in elements],
            [element["reference"] for element in elements],
            [element["metadata"] for element in elements]
        )
    
    @classmethod
    def from_parquet(path: Path) -> None:
        raise NotImplementedError
    
    @classmethod
    def from_s3(path: Path) -> None:
        raise NotImplementedError
    
    @classmethod
    def from_dataset(
        cls, 
        hf_repo: str, 
        split: str = "train",
        original_feature: str = "original",
        reference_feature: str = "reference",
    ) -> None:
        dataset = datasets.load_dataset(hf_repo, split=split)
        return cls(
            originals=dataset[original_feature],
            references=dataset[reference_feature],
            metadata=[{"lang": lang} for lang in dataset["lang"]]
        )
