from pathlib import Path

from pydantic_evals import Dataset

from llm_evals.evaluators import IsJson

from .schemas import CategoryPredictionInput, ExpectedResult, MetaData

DATASET_PATH = Path(__file__).parent / "dataset.yaml"


CUSTOM_EVALUATOR_TYPES = (IsJson,)


dataset = Dataset[CategoryPredictionInput, ExpectedResult, MetaData].from_file(
    path=DATASET_PATH, custom_evaluator_types=CUSTOM_EVALUATOR_TYPES
)


def save_dataset_to_yaml():
    dataset.to_file(
        DATASET_PATH,
        custom_evaluator_types=CUSTOM_EVALUATOR_TYPES,
    )
