from typing import Mapping

from metaflow import FlowSpec, Parameter, step, current
from datasets import load_dataset

from spellcheck.utils import get_logger
from spellcheck.processing import DataProcessor


LOGGER = get_logger("INFO")

class DataProcessing(FlowSpec):
    """Processing pipeline to modify the Spellcheck training dataset algorithmically.
    """

    dataset_hf_repo = Parameter(
        name="dataset_hf_repo",
        default="openfoodfacts/spellcheck-dataset",
        type=str,
        help="Dataset id stored in the Hugging Face OFF repository."
    )

    dataset_revision = Parameter(
        name="dataset_revision",
        default="v3",
        type=str,
        help="Dataset revision indicating the version for processing. Default to v3."
    )

    dataset_version = Parameter(
        name="dataset_version",
        type=str,
        required=True,
        help="New processed dataset version."
    )

    dataset_split = Parameter(
        name="dataset_split",
        type=str,
        required=False,
        default="train+test",
        help="Select split from dataset. Could be 'train', 'test' or 'train+test'",
    )

    dataset_test_size = Parameter(
        name="dataset_test_size",
        type=float,
        required=False,
        default=0.1,
        help="Dataset test split size used during push_to_hub."
    )

    @step
    def start(self):
        """Load dataset"""
        if self.dataset_split not in ["train", "test", "train+test"]:
            raise ValueError("Invalid value for dataset_split. Should be 'train', 'test', or 'train+test'.")
        
        self.dataset = load_dataset(
            self.dataset_hf_repo,
            revision=self.dataset_revision,
            split=self.dataset_split,
        )
        LOGGER.info(f"Dataset loaded:\n{self.dataset}")
        self.next(self.process)
    
    @step
    def process(self):
        """Process dataset."""

        def process_fn(sample: Mapping) -> Mapping:
            """Map function used to process dataset. Add any additional processing in this function.

            Args:
                sample (Mapping): Dataset batch element.

            Return:
                (Mapping): Processed batch.
            """
            processed_labels = DataProcessor.align_oe(
                references=sample["text"],
                texts=sample["label"]
            )
            processed_labels = DataProcessor.align_whitespace_percentage(
                references=sample["text"],
                texts=processed_labels
            )
            sample["label"] = processed_labels
            return sample

        self.processed_dataset = self.dataset.map(process_fn, batched=True)
        LOGGER.info(f"Dataset processed:\n{self.processed_dataset}")
        self.next(self.end)

    @step
    def end(self):
        """End of the process."""
        self.processed_dataset.train_test_split(
            test_size=self.dataset_test_size,
            seed=42,
        ).push_to_hub(
            repo_id=self.dataset_hf_repo,
            commit_message=self.dataset_version,
            commit_description="Metaflow run id:" + current.run_id, # Store Metaflow run id for traceability
        )
        LOGGER.info("Data processing finished succesfully.")


if __name__ == "__main__":
    DataProcessing()
