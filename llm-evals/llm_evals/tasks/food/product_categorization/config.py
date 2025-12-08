from llm_evals.types import TaskConfig

from .datasets import dataset
from .schemas import CategoryPredictionResponseModel
from .tasks import DEFAULT_INSTRUCTIONS, task

CONFIG = TaskConfig(
    dataset=dataset,
    task=task,
    instructions=DEFAULT_INSTRUCTIONS,
    output_type=CategoryPredictionResponseModel,
    multiple_images=True,
)
