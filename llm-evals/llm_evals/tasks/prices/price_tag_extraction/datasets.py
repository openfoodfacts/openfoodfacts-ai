from pathlib import Path

from pydantic_evals import Dataset

from llm_evals.evaluators import IsJson

from .evaluators import CheckExtraction
from .schemas import ExpectedResult, MetaData, PriceTagExtractionInput

DATASET_PATH = Path(__file__).parent / "dataset.yaml"


CUSTOM_EVALUATOR_TYPES = (IsJson, CheckExtraction)


dataset = Dataset[PriceTagExtractionInput, ExpectedResult, MetaData].from_file(
    path=DATASET_PATH, custom_evaluator_types=CUSTOM_EVALUATOR_TYPES
)


def save_dataset_to_yaml():
    dataset.to_file(
        DATASET_PATH,
        custom_evaluator_types=CUSTOM_EVALUATOR_TYPES,
    )
