from pathlib import Path
from typing import TypedDict

from pydantic_evals import Dataset

from llm_evals.evaluators import IsCorrectJsonSchema, IsJson

from .schemas import ExpectedResult

DATASET_PATH = Path(__file__).parent / "dataset.yaml"


class PriceTagExtractionInput(TypedDict):
    image_url: str


class MetaData(TypedDict):
    price_tag_id: int
    tags: list[str] | None


CUSTOM_EVALUATOR_TYPES = (IsJson, IsCorrectJsonSchema)


dataset = Dataset[PriceTagExtractionInput, ExpectedResult, MetaData].from_file(
    path=DATASET_PATH, custom_evaluator_types=CUSTOM_EVALUATOR_TYPES
)


def save_dataset_to_yaml():
    dataset.to_file(
        DATASET_PATH,
        custom_evaluator_types=CUSTOM_EVALUATOR_TYPES,
    )
